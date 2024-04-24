# Inspiré de BB84 example dans QEurope

from PSC_Network import *

# Simulation time (ns)
simtime = 1000000

# Simulation 
ns.sim_reset()

net2 = QNetwork("Polytechnique")
net2.Add_QNode("QNodeParis")
net2.Add_QNode("QNodePalaiseau")
net2.Add_QNode("QNodeMillieu")
net2.Connect_QNode("QNodeParis", "QNodePalaiseau", distance= 50 , linktype="fiber")
net2.Connect_QNode("QNodeMillieu", "QNodePalaiseau", distance= 500 , linktype="satellite",tsat=0.98)
net2.Connect_QNode("QNodeParis", "QNodeMillieu", distance= 500 , linktype="satellite", tsat=0.98)
net = net2.network
Alice = net.get_node("QNodeParis")
Bob = net.get_node("QNodePalaiseau")
Steve =  net.get_node("QNodeMillieu")


Alice.keyout[Steve.name], Bob.keyin[Steve.name] = [], []


# Alice envoie à Bob 
ProtocolS = SendBB84(Steve, QNode_init_succ, QNode_init_flip, Alice)
ProtocolS.start()
#Steve transmet à Bob   

ProtocolM= TransmitProtocol(Alice,Bob,switch_succ,Steve)
ProtocolM.start()
# Bob reçoit et mesure
protocolA = ReceiveProtocol(Steve, QNode_meas_succ, Qnode_meas_flip, True, Bob)
protocolA.start()
addDarkCounts(Bob.keyin[Alice.name], pdarkbest, int(simtime/QNode_init_time))

stat = ns.sim_run(duration = simtime)
Lres = Sifting(Bob.keyin[Steve.name], Alice.keyout[Steve.name])



print("Number of qubits sent by Alice (Qonnector): " + str(len(Alice.keyout[Steve.name])))
print("Number of qubits received by Bob (Qlient): " + str(len(Bob.keyin[Steve.name])))
print("Raw key rate : " + str(len(Bob.keyin[Steve.name])/(simtime*1e-9))+" bits per second")
print("Throughout : "+ str(len(Bob.keyin[Steve.name])/len(Alice.keyout[Steve.name]) )+ " bits per channel use")
print("QBER : "+str(estimQBER(Lres)))


