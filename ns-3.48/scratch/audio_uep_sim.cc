#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"

#include <fstream>
#include <string>
#include <cmath>

// Set up a simple point-to-point network between a server node and a client node using PointToPointHelper
// Attach a RateErrorModel to the receiving node's device to simulate your experimental congestion bounds (7% and 12% packet loss).
// Two separate UDP sockets or an application traffic helper to model the independent QUIC channels 
// Stream 0 (Vocals): Packets sent frequently, heavily padded with extra bytes to simulate the 30% Vocal FEC redundancy ratio (f_v)
// Stream 1 (Instruments): Packets sent with minimal overhead to simulate the 4% Instrumental FEC ratio (f_i).The 
//Timestamp Clock: Attach custom headers to your outgoing packet streams that record the Presentation Time Stamp (PTS = t).


using namespace ns3;

// Creates a named logging component so ns-3 can print debugging messages.
NS_LOG_COMPONENT_DEFINE("AudioUepSimulation");

// Global CSV output stream.
// This lets the packet receive callback write packet-arrival data to a file.
std::ofstream g_outputFile;

// ============================================================================
// CUSTOM HEADER: STORES STREAM ID + PRESENTATION TIME STAMP
// ============================================================================

class QuicStreamHeader : public Header {
public:
    // Registers this custom header type with ns-3's internal TypeId system.
    static TypeId GetTypeId(void) {
        static TypeId tid = TypeId("ns3::QuicStreamHeader")
            .SetParent<Header>()
            .AddConstructor<QuicStreamHeader>();
        return tid;
    }

    // Returns the TypeId for this specific object instance.
    TypeId GetInstanceTypeId(void) const override {
        return GetTypeId();
    }

    // Header size:
    // 1 byte for StreamID
    // 4 bytes for PTS
    // Total = 5 bytes
    uint32_t GetSerializedSize(void) const override {
        return 5;
    }

    // Converts the header fields into raw bytes before transmission.
    void Serialize(Buffer::Iterator start) const override {
        start.WriteU8(m_streamId);       // Writes stream ID: 0 = vocals, 1 = instruments.
        start.WriteHtonU32(m_pts);       // Writes PTS in network byte order.
    }

    // Converts raw received bytes back into usable header fields.
    uint32_t Deserialize(Buffer::Iterator start) override {
        m_streamId = start.ReadU8();      // Reads the stream ID.
        m_pts = start.ReadNtohU32();      // Reads the PTS and converts from network byte order.
        return 5;
    }

    // Prints header contents if ns-3 debugging wants to display this packet.
    void Print(std::ostream &os) const override {
        os << "StreamID=" << static_cast<uint32_t>(m_streamId)
           << " PTS=" << m_pts;
    }

    // Sets whether the packet belongs to the vocal or instrumental stream.
    void SetStreamId(uint8_t id) {
        m_streamId = id;
    }

    // Gets the stream ID.
    uint8_t GetStreamId(void) const {
        return m_streamId;
    }

    // Sets the Presentation Time Stamp.
    // This represents when the audio frame is supposed to play.
    void SetPts(uint32_t pts) {
        m_pts = pts;
    }

    // Gets the Presentation Time Stamp.
    uint32_t GetPts(void) const {
        return m_pts;
    }

private:
    uint8_t m_streamId = 0;   // 0 = vocal object, 1 = instrumental object.
    uint32_t m_pts = 0;       // Audio playback timestamp in milliseconds.
};

// ============================================================================
// PACKET RECEIVE CALLBACK
// ============================================================================

void PacketSinkTrace(Ptr<const Packet> packet, const Address &from) {
    // Gets the current simulator time when the packet arrives.
    // This is treated as T_decode, the time the packet becomes available.
    double tDecodeMs = Simulator::Now().GetMilliSeconds();

    // Makes a copy because packet objects are immutable in callbacks.
    Ptr<Packet> copy = packet->Copy();

    // Creates an empty custom header object.
    QuicStreamHeader header;

    // Reads the custom stream metadata from the received packet.
    if (copy->PeekHeader(header)) {
        // If the output file is open, log this packet's result.
        if (g_outputFile.is_open()) {
            g_outputFile << static_cast<uint32_t>(header.GetStreamId()) << ","
                         << header.GetPts() << ","
                         << tDecodeMs << "\n";
        }
    }
}

// ============================================================================
// MAIN SIMULATION
// ============================================================================

int main(int argc, char *argv[]) {
    // Packet loss rate.
    // Example: 0.07 means 7% random packet loss.
    double lossRate = 0.12;

    // Link bandwidth.
    // This controls the simulated network capacity.
    std::string dataRate = "5Mbps";

    // One-way propagation delay.
    // 50 ms one-way means roughly 100 ms RTT.
    std::string delay = "50ms";

    // CSV file where packet arrival results will be saved.
    std::string outputPath = "simulation_offsets.csv";

    // UEP FEC setting for the vocal object.
    // 0.30 means the vocal stream gets 30% redundancy overhead.
    double vocalFec = 0.30;

    // UEP FEC setting for the instrumental object.
    // 0.04 means the instrumental stream gets 4% redundancy overhead.
    double instFec = 0.04;

    // Base packet payload before redundancy is added.
    uint32_t basePayloadSize = 1000;

    // Total simulated audio segment length.
    // 1000 ms = 1 second of audio frames.
    uint32_t totalDurationMs = 1000;

    // Audio frame interval.
    // A packet is sent every 20 ms for each stream.
    uint32_t frameStepMs = 20;

    // Lets you override parameters from the terminal.
    CommandLine cmd(__FILE__);
    cmd.AddValue("lossRate", "Packet loss rate, e.g. 0.00, 0.07, 0.12", lossRate);
    cmd.AddValue("dataRate", "Point-to-point data rate", dataRate);
    cmd.AddValue("delay", "Point-to-point one-way delay", delay);
    cmd.AddValue("outputPath", "CSV output path", outputPath);
    cmd.AddValue("vocalFec", "Vocal FEC redundancy ratio", vocalFec);
    cmd.AddValue("instFec", "Instrumental FEC redundancy ratio", instFec);
    cmd.Parse(argc, argv);

    // Sets ns-3 time precision.
    Time::SetResolution(Time::NS);

    // Converts FEC rates into actual packet sizes.
    // Vocal packet = 1000 * 1.30 = 1300 bytes.
    uint32_t vocalPayloadSize =
        static_cast<uint32_t>(std::round(basePayloadSize * (1.0 + vocalFec)));

    // Instrument packet = 1000 * 1.04 = 1040 bytes.
    uint32_t instPayloadSize =
        static_cast<uint32_t>(std::round(basePayloadSize * (1.0 + instFec)));

    // Creates two nodes:
    // Node 0 = media server
    // Node 1 = client receiver
    NodeContainer nodes;
    nodes.Create(2);

    // Builds a simple point-to-point network link.
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue(dataRate));
    p2p.SetChannelAttribute("Delay", StringValue(delay));

    // Installs network devices on both nodes and connects them.
    NetDeviceContainer devices = p2p.Install(nodes);

    // Creates random packet loss.
    // This simulates congestion or unreliable wireless conditions.
    Ptr<RateErrorModel> errorModel = CreateObject<RateErrorModel>();
    errorModel->SetAttribute("ErrorRate", DoubleValue(lossRate));
    errorModel->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));

    // Applies packet loss only to packets received by the client.
    devices.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(errorModel));

    // Installs IP networking support on both nodes.
    InternetStackHelper stack;
    stack.Install(nodes);

    // Assigns IPv4 addresses to the point-to-point devices.
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    // Uses separate UDP ports to model independent QUIC-like streams.
    uint16_t vocalPort = 9000;
    uint16_t instPort = 9001;

    // Creates a socket for the vocal stream.
    Ptr<Socket> vocalSocket =
        Socket::CreateSocket(nodes.Get(0), UdpSocketFactory::GetTypeId());

    // Creates a socket for the instrumental stream.
    Ptr<Socket> instSocket =
        Socket::CreateSocket(nodes.Get(0), UdpSocketFactory::GetTypeId());

    // Connects the vocal socket to the client.
    vocalSocket->Connect(InetSocketAddress(interfaces.GetAddress(1), vocalPort));

    // Connects the instrumental socket to the client.
    instSocket->Connect(InetSocketAddress(interfaces.GetAddress(1), instPort));

    // Creates a receiving app for vocal packets.
    PacketSinkHelper vocalSinkHelper(
        "ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), vocalPort)
    );

    // Creates a receiving app for instrumental packets.
    PacketSinkHelper instSinkHelper(
        "ns3::UdpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), instPort)
    );

    // Installs vocal receiver on the client.
    ApplicationContainer vocalSinkApp = vocalSinkHelper.Install(nodes.Get(1));

    // Installs instrumental receiver on the client.
    ApplicationContainer instSinkApp = instSinkHelper.Install(nodes.Get(1));

    // Starts receivers before packets are sent.
    vocalSinkApp.Start(Seconds(1.0));
    instSinkApp.Start(Seconds(1.0));

    // Stops receivers after the simulated stream finishes.
    vocalSinkApp.Stop(Seconds(10.0));
    instSinkApp.Stop(Seconds(10.0));

    // Gets references to the packet sinks.
    Ptr<PacketSink> vocalSink = DynamicCast<PacketSink>(vocalSinkApp.Get(0));
    Ptr<PacketSink> instSink = DynamicCast<PacketSink>(instSinkApp.Get(0));

    // Connects packet arrival events to the logging function.
    vocalSink->TraceConnectWithoutContext("Rx", MakeCallback(&PacketSinkTrace));
    instSink->TraceConnectWithoutContext("Rx", MakeCallback(&PacketSinkTrace));

    // Opens the CSV file for writing packet logs.
    g_outputFile.open(outputPath, std::ios::out);

    // Stops the program if the output file cannot be created.
    if (!g_outputFile.is_open()) {
        NS_LOG_UNCOND("Error opening output CSV.");
        return 1;
    }

    // Writes CSV column names.
    g_outputFile << "StreamID,PTS,T_decode\n";

    // Streaming begins at 2 seconds.
    Time startTime = Seconds(2.0);

    // Sends one vocal packet and one instrumental packet every 20 ms.
    for (uint32_t t = 0; t < totalDurationMs; t += frameStepMs) {
        // Schedules vocal packet transmission.
        Simulator::Schedule(startTime + MilliSeconds(t), [=]() {
            // Creates vocal packet with FEC overhead included.
            Ptr<Packet> packet = Create<Packet>(vocalPayloadSize);

            // Adds metadata saying this is Stream 0 and should play at time t.
            QuicStreamHeader header;
            header.SetStreamId(0);
            header.SetPts(t);

            // Inserts the header into the packet.
            packet->AddHeader(header);

            // Sends the vocal packet.
            vocalSocket->Send(packet);
        });

        // Schedules instrumental packet transmission.
        Simulator::Schedule(startTime + MilliSeconds(t), [=]() {
            // Creates instrumental packet with smaller FEC overhead.
            Ptr<Packet> packet = Create<Packet>(instPayloadSize);

            // Adds metadata saying this is Stream 1 and should play at time t.
            QuicStreamHeader header;
            header.SetStreamId(1);
            header.SetPts(t);

            // Inserts the header into the packet.
            packet->AddHeader(header);

            // Sends the instrumental packet.
            instSocket->Send(packet);
        });
    }

    // Prints simulation parameters to the terminal.
    NS_LOG_UNCOND("Running UEP simulation...");
    NS_LOG_UNCOND("Vocal FEC: " << vocalFec << ", packet size: " << vocalPayloadSize);
    NS_LOG_UNCOND("Instrument FEC: " << instFec << ", packet size: " << instPayloadSize);
    NS_LOG_UNCOND("Loss rate: " << lossRate);

    // Stops all simulator activity at 11 seconds.
    Simulator::Stop(Seconds(11.0));

    // Runs all scheduled network events.
    Simulator::Run();

    // Closes the CSV file.
    g_outputFile.close();

    // Cleans up ns-3 simulation memory.
    Simulator::Destroy();

    // Prints completion message.
    NS_LOG_UNCOND("Simulation complete. Output saved to " << outputPath);

    return 0;
}