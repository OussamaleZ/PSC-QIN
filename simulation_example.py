# Inspiré de BB84 example dans QEurope

from PSC_Network import *

# Simulation time (ms)
simtime = 100000

# Simulation 
ns.sim_reset()

net2 = QNetwork("Polytechnique")
net2.Add_QNode("Paris")
net2.Add_QNode("Palaiseau")
net2.Connect_QNode("Paris", "Palaiseau", distance=20, linktype="fiber")

net = net2.network
Alice = net.get_node("Paris")
Bob = net.get_node("Palaiseau")

Alice.key, Bob.key = [], []

# Alice envoie à Bob 
ProtocolS = SendBB84(Bob, QNode_init_succ, QNode_init_flip, Alice)
ProtocolS.start()

# Bob reçoit et mesure
protocolA = ReceiveProtocol(Alice, QNode_meas_succ, Qnode_meas_flip, True, Bob)
protocolA.start()

# adddarkcounts -> peut être à rajouter

stat = ns.sim_run(duration = simtime)
Lres = Sifting(Bob.key, Alice.key)

print(Bob.key)


print("Number of qubits sent by Alice (Qonnector): " + str(len(Alice.key)))
print("Number of qubits received by Bob (Qlient): " + str(len(Bob.key)))
print("Raw key rate : " + str(len(Bob.key)/(simtime*1e-9))+" bits per second")
print("Throughout : "+ str(len(Bob.key)/len(Alice.key) )+ " bits per channel use")

