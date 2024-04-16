import netsquid as ns 
from netsquid.nodes import Node
from netsquid.nodes.network import Network
from netsquid.components import QuantumChannel, QuantumMemory, ClassicalChannel
from netsquid.components.models.qerrormodels import FibreLossModel, DephaseNoiseModel
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.protocols import NodeProtocol
from netsquid.components.qprocessor import PhysicalInstruction
from netsquid.components.clock import Clock
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel


import netsquid.components.instructions as instr

import random
import numpy as np
import math 
from scipy.stats import bernoulli
from lossmodel import FreeSpaceLossModel, FixedSatelliteLossModel


#QNode parameters


#Network parameters
fiber_coupling = 0.9
fiber_loss = 0.18
fiber_dephasing_rate = 0.02


#Sattelites parameters
Max_Qlient = 5 #Number of simultaneous link that the Qonnector can create 
f_qubit_satt = 80e6 #Qubit creation attempt frequency in MHz
Satellite_init_time = math.ceil(1e9/f_qubit_satt) #time to create |0> in a Satellite node in ns
Satellite_init_succ = 0.008 #Probability that a qubit creation succeeds
Satellite_init_flip = 0

Satellite_meas_succ=0.95 #Probability that a measurement succeeds
Satellite_meas_flip = 1e-5 #Probability that the measurement outcome is flipped by the detectors 


#QNodes parameters
f_qubit_qnode = 80e6 #Qubit creation attempt frequency
QNode_init_time = math.ceil(1e9/f_qubit_qnode) #time to create |0> in a Qlient node in ns
QNode_init_succ = 0.008 #Probability that a a qubit creation succeed
QNode_init_flip = 0#probability that a qubit is flipped at its creation
QNode_meas_succ=0.95 #Probability that a measurement succeeds
Qnode_meas_flip = 1e-5 #Probability that the measurement outcome is flipped by the detectors 

# Common parameters
switch_succ=0.9 #probability that transmitting a qubit from a qlient to another succeeds
BSM_succ = 0.9 #probability that a Bell state measurement of 2 qubits succeeds
EPR_succ=0.01 #probability that an initialisation of an EPR pair succeeds
f_EPR = 80e6 #EPR pair creation attempt frequency in MHz
EPR_time = math.ceil(1e9/f_EPR) # time to create a bell pair in a Qonnector node (ns)
f_GHZ = 8e6 #GHZ state creation attempt frequency in MHz
GHZ3_time = math.ceil(1e9/f_GHZ) #time to create a GHZ3 state (ns)
GHZ3_succ = 2.5e-3 #probability that creating a GHZ3 state succeeds
GHZ4_time = math.ceil(1e9/f_GHZ) #time to create a GHZ4 state (ns)
GHZ4_succ = 3.6e-3 #probability that creating a GHZ4 state succeeds
GHZ5_time = math.ceil(1e9/f_GHZ) #time to create a GHZ5 state (ns)
GHZ5_succ = 9e-5 #probability that creating a GHZ5 state succeeds


# Satellite to Ground channel parameters
txDiv = 5e-6
sigmaPoint = 0.5e-6
rx_aperture_sat = 1
Cn2_sat = 0

# Free space channel parameter
W0 = 1550*1e-9/(txDiv*np.pi)
rx_aperture_drone = 0.4
rx_aperture_ground = 1
Cn2_drone_to_ground = 10e-16#1e-15
Cn2_drone_to_drone = 10e-18
wavelength = 1550*1e-9
c = 299792.458 #speed of light in km/s
Tatm = 1

#Quantum operations accessible to the Satellites
satellites_physical_instructions = [
    PhysicalInstruction(instr.INSTR_INIT, duration=Satellite_init_time),
    PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_S, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_I, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=True),
    PhysicalInstruction(instr.INSTR_MEASURE, duration=1, parallel=True, topology=[0,1]),
    PhysicalInstruction(instr.INSTR_MEASURE_BELL, duration = 1, parallel=True),
    PhysicalInstruction(instr.INSTR_SWAP, duration = 1, parallel=True)
]
#Quantum operations accessible to the QNodes
qlient_physical_instructions = [
    PhysicalInstruction(instr.INSTR_INIT, duration=QNode_init_time),
    PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_S, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_I, duration=1, parallel=True, topology=[0]),
    PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=True),
    PhysicalInstruction(instr.INSTR_MEASURE, duration=1, parallel=False, topology=[0]),
    PhysicalInstruction(instr.INSTR_MEASURE, duration=1, parallel=True, topology=[0,1]),
    PhysicalInstruction(instr.INSTR_MEASURE_BELL, duration = 1, parallel=True),
    PhysicalInstruction(instr.INSTR_SWAP, duration = 1, parallel=True)  
]




#Free space parameters
class Satellite(Node):
    """A Qonnector node
    
    Parameters:
    QNodeList: List of connected QNodes
    QNodePorts: Dictionnary of the form {QNode: [port_to_send, port_to_receive]}
    QNodeKeys : Dictionnary for QKD of the form {QNode: [key]}
    """
    
    def __init__(self, name,QlientList=None,
                  QlientPorts=None,QlientKeys=None):
        super().__init__(name=name)
        self.QlientList = QlientList
        self.QlientPorts = QlientPorts
        self.QlientKeys = QlientKeys


class QNode(Node) :
    """A node in our network with : 
    name 
    neighbourList : liste des voisins 
    phys_instruction: list of physical instructions for the Qlient
    portsDict : dictionnaire sous le forme {voisin : [port_tosend,port_toreceive]}
    portList : liste constitué de deux ports un pour envoyé et un pour recevoir
    key : clé pour BB84 """
    #Ici il faut essayer de bien comprende ce que veulent dire les ports ça peut aider à débugger
    def __init__(self, name, phys_instruction, neighbourList = None, portsDict = None, portsList = None, key = None):
        super().__init__(name = name)
        self.neighbourList = neighbourList 
        self.portsDict = portsDict  #
        self.portsList = portsList 
        self.key = key 
        
        #On rajoute une mémoire quantique, le premier paramètre réfère au nim (str) de la mémoire quantique,
        # le deuxième est le nombre de memoire quantique disponible (ici on a besoin d'une mémoire quantique pour chaque lien et donc on a besoin de deux mémoires quantiques)
        # le troisième paramètre indique la liste des physical instructions
        #qmem ne sert plus à rien. qmem avait pour rôle d'optimiser le code car le Qlient n'est attaché qu'à un Qonnector et donc  n'avait besoin que d'une mémoire quantique. 
        #En revanche, dans notre modèle le QNode peut être lié à plusieurs Qonnectors et donc a besoin de plusieurs mémoires quantiques. Pour cela on doit utiliser la méthode add_subcompenent("la mémoire qu'on veut ")
        #qmem = QuantumProcessor("QNodeMemory{}".format(name), num_positions=2, phys_instructions=phys_instruction) 

        #self.qmemory = qmem 


class QNetwork():
    """Le réseau"""

    def __init__(self, name):
        self.network = Network(name)
        self.name = name

    def Add_QNode(self, name):
        """Méthode afin d'ajouter un noeud au réseau"""
        noeud = QNode(name, phys_instruction= qlient_physical_instructions, neighbourList = [], portsDict = {}, portsList =[], key = None)
        self.network.add_node(noeud)

    def Connect_QNode(self, qnode1, qnode2, distance, dist_sat1 = None, dist_sat2 = None, tsat1 = None, tsat2 = None, linktype = "fiber"):
        """Relie QNode1 et QNode2 avec avec une fibre
        ##Parameters##
        qnode1 : (str) nom du premier node 
        qnode2 : (str) nom du deuxième node
        distance : distance entre les deux nodes si le lien est fibré
        dist_sat1 : distance entre QNode1 et le satellite si le lien est satellitaire
        dist_sat2 : distance entre QNode2 et le satellite si le lien est satellitaire
        tsat1 : transmittance atmosphérique entre QNode1 et le satellite si le lien est satellitaire
        tsat2 : transmittance atmosphérique entre QNode2 et le satellite si le lien est satellitaire
        linktype : type de lien entre les deux noeuds soit "fiber" ou "satellite"
        """ 

        network = self.network
        QNode1 = network.get_node(qnode1)
        QNode2 = network.get_node(qnode2)
        if linktype == "fiber":
            #QuantumeChannel(name (str),
            #                 delay : fixed transmission to use if delay_model is None en ns
            #                 length : longueur du Channel en Km
            #                  models : dictionnaire des modèles qu'on va utiliser)
           
            qchannel12 = QuantumChannel("QuantumChannel{}".format(QNode1.name) + "to{}".format(QNode2.name), delay = 1, length = distance,
                                        models = {"quantumlossmodel" : FibreLossModel(p_loss_init = 1 - fiber_coupling, p_loss_length = fiber_loss),
                                                "quantumnoisemodel" : DephaseNoiseModel(dephase_rate = fiber_dephasing_rate, time_independent = True)})
            qchannel21 = QuantumChannel("QuantumChannel{}".format(QNode2.name) + "to{}".format(QNode1.name), delay = 1, length = distance,
                                        models = {"quantumlossmodel" : FibreLossModel(p_loss_init = 1 - fiber_coupling, p_loss_length = fiber_loss),
                                                "quantumnoisemodel" : DephaseNoiseModel(dephase_rate = fiber_dephasing_rate, time_independent = True)})
           
            #add_connection ( (Node) premier node à connecter,
            #                 (Node) deuxième node à connecter,
            #                 (Channel) où va être placé la connection du noeud 1 au noeud 2)
            # retourne deux (str) qui sont les noms des deux ports
            QNode1_send, QNode2_receive = network.add_connection(
            QNode1, QNode2, channel_to = qchannel12, label = "quantum{}".format(QNode1.name) + "to{}".format(QNode2.name)
            )

            QNode2_send, QNode1_receive = network.add_connection(
                QNode2, QNode1, channel_to = qchannel21, label = "quantum{}".format(QNode2.name) + "to{}".format(QNode1.name)
            )

           #QuantumProcessor ((str) name,
           #                   num position : numéro de quantum memory availabl (ici pn choisit 2 donc une pour envoyer et une pour recevoir)
           #                   phys_instructions : les physicals instructions qu'on a le droits d'utiliser)


            qmem1 = QuantumProcessor( "QNodeMemoryTo{}".format(QNode2.name), num_positions=2, phys_instructions = qlient_physical_instructions)
           #subcompenent est un attribut de la class Node qui est une liste de components
            QNode1.add_subcomponent(qmem1)
            QNode1.neighbourList.append(QNode2)
            QNode1.portsDict[qnode2] = [QNode1_send,QNode1_receive]

            qmem2 = QuantumProcessor( "QNodeMemoryTo{}".format(QNode1.name), num_positions=2, phys_instructions = qlient_physical_instructions)
            QNode2.add_subcomponent(qmem2)
            QNode2.neighbourList.append(QNode1)
            QNode2.portsDict[qnode1] = [QNode2_send,QNode2_receive]
            
            #We start by QNode1
            #C'est un message qui va être envoyé ; je pense qu'il faut mieux expliquer
            def route_qubits1(msg):
                #target ici est une qmem
                target = msg.meta.pop('internal', None)
                if isinstance(target, QuantumMemory):
                    if not target.has_supercomponent(QNode1): #ie on espere que ce soit une qmemoire de QNode1
                        raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                    target.ports['qin'].tx_input(msg)
                else:
                    QNode1.ports[QNode1_send].tx_output(msg)

            # Connect the Qonnector's ports
            qmem1.ports['qout'].bind_output_handler(route_qubits1) #port to send to Qlient
            QNode1.ports[QNode1_receive].forward_input(qmem1.ports["qin"]) #port to receive from Qlient
        

            # Connect the Qlient's ports 
            #QNode2.ports[QNode2_receive].forward_input(qmem2.ports["qin"]) #port to receive from qonnector
            #qmem2.ports["qout"].forward_output(QNode2.ports[QNode2_send]) #port to send to qonnector
            
            #We do the same for QNode2
            def route_qubits2(msg):
                target = msg.meta.pop('internal', None)
                if isinstance(target, QuantumMemory):
                    if not target.has_supercomponent(QNode2):
                        raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                    target.ports['qin'].tx_input(msg)
                else:
                    QNode2.ports[QNode2_send].tx_output(msg)

            # Connect the Qonnector's ports

            qmem2.ports['qout'].bind_output_handler(route_qubits2) #port to send to Qlient
            QNode2.ports[QNode2_receive].forward_input(qmem2.ports["qin"]) #port to receive from Qlient
        

            # Connect the Qlient's ports 
            #QNode1.ports[QNode1_receive].forward_input(qmem1.ports["qin"]) #port to receive from qonnector
            #qmem1.ports["qout"].forward_output(QNode1.ports[QNode1_send]) #port to send to qonnector


            cchannel12 = ClassicalChannel("ClassicalChannel{}".format(QNode1.name) + "to{}".format(QNode2.name), delay = 1, length = distance)
            cchannel21 = ClassicalChannel("ClassicalChannel{}".format(QNode2.name) + "to{}".format(QNode1.name), delay = 1, length = distance)

            network.add_connection(QNode1, QNode2, channel_to = cchannel21, label="Classical{}".format(QNode2.name) + "to{}".format(QNode1.name), 
                                port_name_node1="cout_{}".format(QNode2.name), port_name_node2="cin_{}".format(QNode1.name))
            network.add_connection(QNode2, QNode1, channel_to=cchannel12, label = "Classical{}".format(QNode1.name) + "to{}".format(QNode2.name),
                                port_name_node1="cout_{}".format(QNode1.name), port_name_node2="cin_{}".format(QNode2.name))
            
        
        elif linktype == "satellite":
            # à modfier 

            #create dedicated quantum memories at each qonnector
            QN1 = network.get_node(QNode1)
            QN2 = network.get_node(QNode2)
            #Create a satellite node with a quantum processor for each QNode
            Satellite = Satellite  ("Satellite{}".format(QNode1 + QNode2), QlientList=[],QlientPorts={},QlientKeys={})
            network.add_node(Satellite) 
            qmem3 = QuantumProcessor( "SatelliteMemoryTo{}".format(QNode1), num_positions=2 ,
                                phys_instructions=satellites_physical_instructions)
            Satellite.add_subcomponent(qmem3)
            qmem4 = QuantumProcessor( "SatelliteMemoryTo{}".format(QNode2), num_positions=2 ,
                                phys_instructions=satellites_physical_instructions)
            Satellite.add_subcomponent(qmem4)
            
            
            qmem1 = QuantumProcessor( "SatelliteMemoryTo{}".format(Satellite.name), num_positions=2 ,
                                phys_instructions=satellites_physical_instructions)
            QN1.add_subcomponent(qmem1)
            qmem2 = QuantumProcessor( "SatelliteMemoryTo{}".format(Satellite.name), num_positions=2 ,
                                phys_instructions=satellites_physical_instructions)
            QN2.add_subcomponent(qmem2)
            
            
            #Connect Satellite with QN1 (only downlink)
            qchannel1 = QuantumChannel("SatChannelto{}".format(QNode1),length=dist_sat1, delay=1,
                                   models={"quantum_loss_model": FixedSatelliteLossModel(txDiv, sigmaPoint,
                                                                            rx_aperture_sat, Cn2_sat, wavelength,tsat1)})
            qchannel3 = QuantumChannel("SatChannelto{}".format(Satellite),length=dist_sat1, delay=1,
                                   models={"quantum_loss_model": FixedSatelliteLossModel(txDiv, sigmaPoint,
                                                                            rx_aperture_sat, Cn2_sat, wavelength,tsat1)})
            #connect the channels to nodes
            Sat1_send, QN1_receive = network.add_connection(
                    Satellite, QN1, channel_to=qchannel1, label="SatelliteChanTo{}".format(QNode1))
            QN1_send, Sat1_rec = network.add_connection(
                    QN1, Satellite, channel_to=qchannel3, label="SatelliteChanTo{}".format(Satellite))

        
            #update both node properties
            Satellite.QlientList.append(QNode1)
            Satellite.QlientPorts[QNode1] = [Sat1_send]
            Satellite.QlientKeys[QNode1] = []
        
            QN1.QlientList.append(Satellite.name)
            QN1.QlientPorts[Satellite.name] = [QN1_send,QN1_receive]
            QN1.QlientKeys[Satellite.name] = []
        
            # Connect the Satellite and Qonnector's ports
            def route_qubits3(msg):
                target = msg.meta.pop('internal', None)

                if isinstance(target, QuantumMemory):
                    if not target.has_supercomponent(Satellite):
                        raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                    target.ports['qin'].tx_input(msg)
                else:
                    Satellite.ports[Sat1_send].tx_output(msg)
            
        
            qmem3.ports['qout'].bind_output_handler(route_qubits3) 
            
            def route_qubits5(msg):
                target = msg.meta.pop('internal', None)

                if isinstance(target, QuantumMemory):
                    if not target.has_supercomponent(QN1):
                        raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                    target.ports['qin'].tx_input(msg)
                else:
                    QN1.ports[QN1_send].tx_output(msg)
                    
            qmem1.ports['qout'].bind_output_handler(route_qubits5) 
            QN1.ports[QN1_receive].forward_input(qmem1.ports["qin"]) 
        
            #Classical channels on top of that
            cchannel1 = ClassicalChannel("ClassicalChannelto{}".format(QNode1),length=dist_sat1, delay=1)
            cchannel2 = ClassicalChannel("ClassicalChanneltoSatellite",length=dist_sat1, delay=1)
        
            network.add_connection(Satellite, QN1, channel_to=cchannel1, 
                               label="Classicalto{}".format(QNode1), port_name_node1="cout_{}".format(QNode1),
                               port_name_node2="cin_{}".format(Satellite.name))
            network.add_connection(QN1, Satellite, channel_to=cchannel2, 
                               label="ClassicaltoSat".format(QNode1), port_name_node1="cout_{}".format(Satellite.name),
                               port_name_node2="cin_{}".format(QNode1))
            
            #Do the same with Qonn2
            qchannel2 = QuantumChannel("SatChannelto{}".format(QN2),length=dist_sat2, delay=1,
                                   models={"quantum_loss_model": FixedSatelliteLossModel(txDiv, sigmaPoint,
                                                                            rx_aperture_sat, Cn2_sat, wavelength,tsat2)})
            qchannel4 = QuantumChannel("SatChannelto{}".format(Satellite),length=dist_sat2, delay=1,
                                   models={"quantum_loss_model": FixedSatelliteLossModel(txDiv, sigmaPoint,
                                                                            rx_aperture_sat, Cn2_sat, wavelength,tsat2)})        
            #connect the channels to nodes
            Sat2_send, QN2_receive = network.add_connection(
                    Satellite, QN2, channel_to=qchannel2, label="SatelliteChanTo{}".format(QN2))
            QN2_send, Sat2_receive = network.add_connection(
                    QN2, Satellite, channel_to=qchannel4, label="SatelliteChanTo{}".format(Satellite))

        
            #update both node properties
            Satellite.QlientList.append(QN2)
            Satellite.QlientPorts[QN2] = [Sat2_send]
            Satellite.QlientKeys[QN2] = []
        
            QN2.QlientList.append(Satellite.name)
            QN2.QlientPorts[Satellite.name] = [QN2_send,QN2_receive]
            QN2.QlientKeys[Satellite.name] = []
        
            # Connect the Satellite and Qonnector's ports
            def route_qubits4(msg):
                target = msg.meta.pop('internal', None)

                if isinstance(target, QuantumMemory):
                    if not target.has_supercomponent(Satellite):
                        raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                    target.ports['qin'].tx_input(msg)
                else:
                    Satellite.ports[Sat2_send].tx_output(msg)
            
        
            qmem4.ports['qout'].bind_output_handler(route_qubits4) 
            
            def route_qubits6(msg):
                target = msg.meta.pop('internal', None)

                if isinstance(target, QuantumMemory):
                    if not target.has_supercomponent(QN2):
                        raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                    target.ports['qin'].tx_input(msg)
                else:
                    QN2.ports[QN2_send].tx_output(msg)
                    
            qmem2.ports['qout'].bind_output_handler(route_qubits6) 
        
             
            QN2.ports[QN2_receive].forward_input(qmem2.ports["qin"]) 
        
            #Classical channels on top of that
            cchannel3 = ClassicalChannel("ClassicalChannelto{}".format(QNode2),length=dist_sat2, delay=1)
            cchannel4 = ClassicalChannel("ClassicalChanneltoSatellite",length=dist_sat2, delay=1)
        
            network.add_connection(Satellite, QN2, channel_to=cchannel3, 
                               label="Classicalto{}".format(QNode1), port_name_node1="cout_{}".format(QNode2),
                               port_name_node2="cin_{}".format(Satellite.name))
            network.add_connection(QN2, Satellite, channel_to=cchannel4, 
                               label="ClassicaltoSat".format(QNode2), port_name_node1="cout_{}".format(Satellite.name),
                               port_name_node2="cin_{}".format(QNode2))
            

        else:
            raise ValueError("Mauvais type de lien")

            
        

class ReceiveProtocol(NodeProtocol):
    """Protocole effectué par un noeud lors de la réception d'un qubit afin de le mesurer"""
    def __init__(self, othernode, measurement_succ, measurement_flip, BB84, node):
        super().__init__(node = node)
        self._othernode = othernode #L'autre bout de communication de type QNode
        self._measurement_succ = measurement_succ # probabilité 
        self._BB84 = BB84 # boolean qui indique si on utilise ou pas le protocole BB84
        self._measurement_flip = measurement_flip # probabilité d'un bitflip
    
    def run(self):
        mem = self.node.subcomponents["QNodeMemoryTo{}".format(self._othernode.name)]
        print(mem)
        port = self.node.ports[self.node.portsDict[self._othernode.name][1]]
        while True :
            yield self.await_port_input(port)
            t = self.node.ports["cin_{}".format(self._othernode.name)].rx_input()       
            b = bernoulli.rvs(self._measurement_succ)
            if b ==1 :
                if self._BB84: #in case we perform BB84
                    base = bernoulli.rvs(0.5) #choose a random basis
                    if base < 0.5:
                        mem.execute_instruction(instr.INSTR_H, [0], physical = False)
                        base = "plusmoins"
                    else:
                        mem.execute_instruction(instr.INSTR_I, [0],physical = False)
                        base = "zeroun"
                else:
                    base = None 
                #print(not(mem.busy))          
                if not(mem.busy):  
                    #on execute la première instruction et on stocke le résultat dans m["M1"][0]
                    m,_,_ = mem.execute_instruction(instr.INSTR_MEASURE,[0],output_key="M1") 
                    yield self.await_program(mem,await_done=True,await_fail=True) # Hamza : J'ai pas bien compris
                    flip = bernoulli.rvs(self._measurement_flip)
                    if (flip==1):
                        if m['M1'][0]==0:
                            m['M1'][0] =1
                        elif m['M1'][0]==1:
                            m['M1'][0]=0
                    #Hamza : on est toujours dans ce cas car on utilise toujours BB84                
                    if m['M1'] is not None and t is not None and base is not None:
                        self.node.key.append(([t.items[0], base],m['M1'][0]))
                            
                    elif m['M1'] is not None and t is not None:
                        self.node.key.append((t.items,m['M1'][0]))
                        
                    elif m['M1'] is not None:
                        self.node.key.append(m['M1'][0])
                            
            mem.reset()



class SendBB84(NodeProtocol):
    
    """Protocol performed by a node to send a random BB84 qubit |0>, |1>, |+> or |-> .
    
    Parameters:
     othernode: name of the receiving node (str).
     init_succ: probability that a qubit creation attempt succeeds.
     init_flip : probability that a qubit created is flipped before the sending.
     """
    
    def __init__(self,othernode, init_succ, init_flip,node):
        super().__init__(node=node)
        self._othernode = othernode # Destination du qubit
        self._init_succ = init_succ # Probabilité de succès de création du qubit
        self._init_flip = init_flip # Probabilité d'un bit flip initial
    
    def run(self):    

        mem = self.node.subcomponents["QNodeMemoryTo{}".format(self._othernode.name)]
            
        clock = Clock(name="clock", start_delay=0, 
                      models={"timing_model": FixedDelayModel(delay=QNode_init_time)})
        self.node.add_subcomponent(clock)
        clock.start() # On regarde la montre
        
        while True:
            mem.reset() # Réinitialisation de la mémoire quantique à chaque étape

            mem.execute_instruction(instr.INSTR_INIT,[0]) # Création du qubit
            yield self.await_program(mem,await_done=True,await_fail=True)
            #print("qubit created")
            succ = bernoulli.rvs(self._init_succ) 
            if (succ == 1):                    
                flip = bernoulli.rvs(self._init_flip)
                if (flip == 1): # Bitflip eventuel 
                    mem.execute_instruction(instr.INSTR_X, [0], physical = False) 
            
                base = bernoulli.rvs(0.5) # Choix aléatoire de la base
                if base <0.5:
                    mem.execute_instruction(instr.INSTR_H,[0])
                    base = "plusmoins"
                else:
                    mem.execute_instruction(instr.INSTR_I,[0])
                    base = "zeroun"
                
                yield self.await_program(mem,await_done=True,await_fail=True)
                
                t = clock.num_ticks
                bit = bernoulli.rvs(0.5) # Choix aléatoire d'un bit
                if bit < 0.5:
                    mem.execute_instruction(instr.INSTR_I, [0], physical=False)
                    self.node.key.append(([t,base],0))
                else:
                    if base == "zeroun":
                        mem.execute_instruction(instr.INSTR_X, [0], physical=False)
                    elif base == "plusmoins":
                        mem.execute_instruction(instr.INSTR_Z, [0], physical=False)
                    self.node.key.append(([t,base],1)) # Ajout à la clé de chiffrement 
                
                qubit, = mem.pop([0])
                self.node.ports["cout_{}".format(self._othernode.name)].tx_output(t) # Envoi du qubit à othernode




def Sifting(Lalice, Lbob):
    """Sifting function to get a list of matching received qubit. If BB84 then the resulting list contains 
    the qubits that were sent and measured in the same basis. If EPR then the resulting list contains the qubit 
    measured by Alice and Bob that came from the same EPR pair
     
     Parameters:
     Lalice, Lbob: lists of outcomes """
    Lres = []
    for i in range(len(Lalice)):
        ta, ma = Lalice[i]
        for j in range(len(Lbob)):
            tb, mb = Lbob[j]
            if ta == tb:
                Lres.append((ma,mb))
        
    return Lres