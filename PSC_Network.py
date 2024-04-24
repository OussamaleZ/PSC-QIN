import netsquid as ns

import netsquid.components.instructions as instr
import netsquid.components.qprogram as qprog
import random 
from scipy.stats import bernoulli
import logging
import math
import numpy as np

from netsquid.components import Channel, QuantumChannel, QuantumMemory, ClassicalChannel
from netsquid.components.models.qerrormodels import FibreLossModel, DepolarNoiseModel, DephaseNoiseModel
from netsquid.nodes import Node, DirectConnection
from netsquid.nodes.connections import Connection
from netsquid.protocols import NodeProtocol
from netsquid.components.models import DelayModel
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components import QuantumMemory
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.nodes.network import Network
from netsquid.qubits import ketstates as ks
from netsquid.protocols.protocol import Signals
from netsquid.components.qprocessor import PhysicalInstruction
from netsquid.qubits import qubitapi as qapi
from netsquid.components.clock import Clock
from netsquid.qubits.dmtools import DenseDMRepr

import sys
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


#Dark Counts parameter
DCRateBest = 100
DCRateWorst = 1000
DetectGateBest = 1e-10
DetectGateWorst = 5e-10
pdarkbest = DCRateBest*DetectGateBest
pdarkworst = DCRateWorst*DetectGateWorst


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
class Satellite_Node(Node):
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
    keyout : dictionnaire des clés envoyées sous la fomeme {voisin : [clé envoyée à ce voisin]}
    keyin : dictionnaire des clés reçues sous la forme {voisin : [clé reçue par ce voisin]} """
    #Ici il faut essayer de bien comprende ce que veulent dire les ports ça peut aider à débugger
    def __init__(self, name, phys_instruction, neighbourList = None, portsDict = None, portsList = None, keyin = None,keyout = None):
        super().__init__(name = name)
        self.neighbourList = neighbourList 
        self.portsDict = portsDict  #
        self.portsList = portsList 
        self.keyout = keyout
        self.keyin = keyin
        
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
        noeud = QNode(name, phys_instruction= qlient_physical_instructions, neighbourList = [], portsDict = {}, portsList =[], keyin = {},keyout={} )
        self.network.add_node(noeud)

    def Connect_QNode(self, qnode1, qnode2, distance=0,  tsat = None,  linktype = "fiber"):
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
        #print(distance)
        if linktype == "fiber":
            #QuantumeChannel(name (str),
            #                 delay : fixed transmission to use if delay_model is None en ns
            #                 length : longueur du Channel en Km
            #                  models : dictionnaire des modèles qu'on va utiliser)
            fiber_speed = 200000  # speed of light in fiber in km/s
            delay = distance / fiber_speed  # delay in seconds
            delay_ns = delay * 1e9  # convert delay to nanoseconds

            # "delay_model" : FibreDelayModel()
           
            qchannel12 = QuantumChannel("QuantumChannel{}".format(qnode1) + "to{}".format(qnode2),  length = distance,delay=1,
                                        models = {"quantum_loss_model" : FibreLossModel(p_loss_init = 1 - fiber_coupling, p_loss_length = fiber_loss),
                                                "quantum_noise_model" : DephaseNoiseModel(dephase_rate = fiber_dephasing_rate, time_independent = True)})
            qchannel21 = QuantumChannel("QuantumChannel{}".format(qnode2) + "to{}".format(qnode1),  length = distance, delay = 1,
                                        models = {"quantum_loss_model" : FibreLossModel(p_loss_init = 1 - fiber_coupling, p_loss_length = fiber_loss),
                                                "quantum_noise_model" : DephaseNoiseModel(dephase_rate = fiber_dephasing_rate, time_independent = True)})
            #print(distance)
            #add_connection ( (Node) premier node à connecter,
            #                 (Node) deuxième node à connecter,
            #                 (Channel) où va être placé la connection du noeud 1 au noeud 2)
            # retourne deux (str) qui sont les noms des deux ports
            QNode1_send, QNode2_receive = network.add_connection(
            qnode1, qnode2, channel_to = qchannel12, label = "quantum{}".format(qnode1) + "to{}".format(qnode2)
            )

            QNode2_send, QNode1_receive = network.add_connection(
                qnode2, qnode1, channel_to = qchannel21, label = "quantum{}".format(qnode2) + "to{}".format(qnode1)
            )

           #QuantumProcessor ((str) name,
           #                   num position : numéro de quantum memory availabl (ici pn choisit 2 donc une pour envoyer et une pour recevoir)
           #                   phys_instructions : les physicals instructions qu'on a le droits d'utiliser)


            qmem1 = QuantumProcessor("QNodeMemoryTo{}".format(qnode2), num_positions=2, phys_instructions = qlient_physical_instructions)
            #subcompenent est un attribut de la class Node qui est une liste de components
            QNode1.add_subcomponent(qmem1)
            QNode1.neighbourList.append(qnode2)
            QNode1.portsDict[qnode2] = [QNode1_send,QNode1_receive]
            QNode1.keyin[qnode2]=[]
            QNode1.keyout[qnode2]=[]


            qmem2 = QuantumProcessor("QNodeMemoryTo{}".format(qnode1), num_positions=2, phys_instructions = qlient_physical_instructions)
            QNode2.add_subcomponent(qmem2)
            QNode2.neighbourList.append(qnode1)
            QNode2.portsDict[qnode1] = [QNode2_send,QNode2_receive]
            QNode2.keyout[qnode1]=[]
            QNode2.keyin[qnode1]=[]


            #We start by QNode1
            #C'est un message qui va être envoyé ; je pense qu'il faut mieux expliquer
            def route_qubits1(msg):
                #target ici est une qmem
                target = msg.meta.pop('internal', None)
                if isinstance(target, QuantumMemory):
                    if not target.has_supercomponent(QNode1): #ie on espere que ce soit une qmemoire de QNode1
                        raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                    #print(target.ports["qin"])
                    target.ports['qin'].tx_input(msg)
                else:
                    QNode1.ports[QNode1_send].tx_output(msg)

            # Connect the Qonnector's ports
            qmem1.ports['qout'].bind_output_handler(route_qubits1) #port to send to Qlient
            QNode1.ports[QNode1_receive].forward_input(qmem1.ports["qin"]) #port to receive from Qlient
        

            # Connect the Qlient's ports 
            QNode2.ports[QNode2_receive].forward_input(qmem2.ports["qin"]) #port to receive from qonnector
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
            #QNode2.ports[QNode2_receive].forward_input(qmem2.ports["qin"]) #port to receive from Qlient
        

            # Connect the Qlient's ports 
            #QNode1.ports[QNode1_receive].forward_input(qmem1.ports["qin"]) #port to receive from qonnector
            #qmem1.ports["qout"].forward_output(QNode1.ports[QNode1_send]) #port to send to qonnector


            cchannel12 = ClassicalChannel("ClassicalChannel{}".format(qnode1) + "to{}".format(qnode2),  length = distance,delay=1)
            cchannel21 = ClassicalChannel("ClassicalChannel{}".format(qnode2) + "to{}".format(qnode1),  length = distance,delay=1)

            network.add_connection(qnode1, qnode2, channel_to = cchannel21, label="Classical{}".format(QNode2.name) + "to{}".format(QNode1.name), 
                                port_name_node1="cout_{}".format(qnode2), port_name_node2="cin_{}".format(qnode1))
            network.add_connection(qnode2, qnode1, channel_to=cchannel12, label = "Classical{}".format(QNode1.name) + "to{}".format(QNode2.name),
                                port_name_node1="cout_{}".format(qnode1), port_name_node2="cin_{}".format(qnode2))
            #print(QNode1)
        
        elif linktype == "satellite":
            #QuantumeChannel(name (str),
            #                 delay : fixed transmission to use if delay_model is None en ns
            #                 length : longueur du Channel en Km
            #                  models : dictionnaire des modèles qu'on va utiliser)
            fiber_speed = 200000  # speed of light in fiber in km/s
            delay = distance / fiber_speed  # delay in seconds
            delay_ns = delay * 1e9  # convert delay to nanoseconds

            # "delay_model" : FibreDelayModel()
           
            qchannel12 = QuantumChannel("QuantumChannel{}".format(qnode1) + "to{}".format(qnode2),  length = distance,delay=1,
                                        models = {"quantum_loss_model" : FixedSatelliteLossModel(txDiv, sigmaPoint,
                                                                            rx_aperture_sat, Cn2_sat, wavelength,tsat),
                                                "quantum_noise_model" : DephaseNoiseModel(dephase_rate = fiber_dephasing_rate, time_independent = True)})
            qchannel21 = QuantumChannel("QuantumChannel{}".format(qnode2) + "to{}".format(qnode1),  length = distance, delay = 1,
                                        models = {"quantum_loss_model" :  FixedSatelliteLossModel(txDiv, sigmaPoint,
                                                                            rx_aperture_sat, Cn2_sat, wavelength,tsat),
                                                "quantum_noise_model" : DephaseNoiseModel(dephase_rate = fiber_dephasing_rate, time_independent = True)})
            #print(distance)
            #add_connection ( (Node) premier node à connecter,
            #                 (Node) deuxième node à connecter,
            #                 (Channel) où va être placé la connection du noeud 1 au noeud 2)
            # retourne deux (str) qui sont les noms des deux ports
            QNode1_send, QNode2_receive = network.add_connection(
            qnode1, qnode2, channel_to = qchannel12, label = "quantum{}".format(qnode1) + "to{}".format(qnode2)
            )

            QNode2_send, QNode1_receive = network.add_connection(
                qnode2, qnode1, channel_to = qchannel21, label = "quantum{}".format(qnode2) + "to{}".format(qnode1)
            )

           #QuantumProcessor ((str) name,
           #                   num position : numéro de quantum memory availabl (ici pn choisit 2 donc une pour envoyer et une pour recevoir)
           #                   phys_instructions : les physicals instructions qu'on a le droits d'utiliser)


            qmem1 = QuantumProcessor("QNodeMemoryTo{}".format(qnode2), num_positions=2, phys_instructions = qlient_physical_instructions)
            #subcompenent est un attribut de la class Node qui est une liste de components
            QNode1.add_subcomponent(qmem1)
            QNode1.neighbourList.append(qnode2)
            QNode1.portsDict[qnode2] = [QNode1_send,QNode1_receive]
            QNode1.keyin[qnode2]=[]
            QNode1.keyout[qnode2]=[]


            qmem2 = QuantumProcessor("QNodeMemoryTo{}".format(qnode1), num_positions=2, phys_instructions = qlient_physical_instructions)
            QNode2.add_subcomponent(qmem2)
            QNode2.neighbourList.append(qnode1)
            QNode2.portsDict[qnode1] = [QNode2_send,QNode2_receive]
            QNode2.keyout[qnode1]=[]
            QNode2.keyin[qnode1]=[]


            #We start by QNode1
            #C'est un message qui va être envoyé ; je pense qu'il faut mieux expliquer
            def route_qubits1(msg):
                #target ici est une qmem
                target = msg.meta.pop('internal', None)
                if isinstance(target, QuantumMemory):
                    if not target.has_supercomponent(QNode1): #ie on espere que ce soit une qmemoire de QNode1
                        raise ValueError("Can't internally route to a quantummemory that is not a subcomponent.")
                    #print(target.ports["qin"])
                    target.ports['qin'].tx_input(msg)
                else:
                    QNode1.ports[QNode1_send].tx_output(msg)

            # Connect the Qonnector's ports
            qmem1.ports['qout'].bind_output_handler(route_qubits1) #port to send to Qlient
            QNode1.ports[QNode1_receive].forward_input(qmem1.ports["qin"]) #port to receive from Qlient
        

            # Connect the Qlient's ports 
            QNode2.ports[QNode2_receive].forward_input(qmem2.ports["qin"]) #port to receive from qonnector
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
            #QNode2.ports[QNode2_receive].forward_input(qmem2.ports["qin"]) #port to receive from Qlient
        

            # Connect the Qlient's ports 
            #QNode1.ports[QNode1_receive].forward_input(qmem1.ports["qin"]) #port to receive from qonnector
            #qmem1.ports["qout"].forward_output(QNode1.ports[QNode1_send]) #port to send to qonnector


            cchannel12 = ClassicalChannel("ClassicalChannel{}".format(qnode1) + "to{}".format(qnode2),  length = distance,delay=1)
            cchannel21 = ClassicalChannel("ClassicalChannel{}".format(qnode2) + "to{}".format(qnode1),  length = distance,delay=1)

            network.add_connection(qnode1, qnode2, channel_to = cchannel21, label="Classical{}".format(QNode2.name) + "to{}".format(QNode1.name), 
                                port_name_node1="cout_{}".format(qnode2), port_name_node2="cin_{}".format(qnode1))
            network.add_connection(qnode2, qnode1, channel_to=cchannel12, label = "Classical{}".format(QNode1.name) + "to{}".format(QNode2.name),
                                port_name_node1="cout_{}".format(qnode1), port_name_node2="cin_{}".format(qnode2))
            #print(QNode1)

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
        #print("mem : ")
        print(mem)
        port = self.node.ports[self.node.portsDict[self._othernode.name][1]]
        #print(port.name)
        #print("3")
        while True :
            yield self.await_port_input(port)
            t = self.node.ports["cin_{}".format(self._othernode.name)].rx_input()  
            #print(self.node.ports["cin_{}".format(self._othernode.name)])
            #print (t)     
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
                if (not(mem.busy)) :
                    #on execute la première instruction et on stocke le résultat dans m["M1"][0]
                    m,_,_ = mem.execute_instruction(instr.INSTR_MEASURE,[0],output_key="M1") 
                    yield self.await_program(mem,await_done=True,await_fail=True) # Hamza : J'ai pas bien compris
                    flip = bernoulli.rvs(self._measurement_flip)
                    if (flip==1):
                        if m['M1'][0]==0:
                            m['M1'][0] =1
                        elif m['M1'][0]==1:
                            m['M1'][0]=0
                    if m['M1'] is not None and t is not None and base is not None:
                        self.node.keyin[self._othernode.name].append(([t.items[0], base],m['M1'][0]))
                                
                    elif m['M1'] is not None and t is not None:
                        self.node.keyin[self._othernode.name].append((t.items,m['M1'][0]))
                            
                    elif m['M1'] is not None:
                        self.node.keyin[self._othernode.name].append(m['M1'][0])
                            
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
                    self.node.keyout[self._othernode.name].append(([t,base],0))
                else:
                    if base == "zeroun":
                        mem.execute_instruction(instr.INSTR_X, [0], physical=False)
                    elif base == "plusmoins":
                        mem.execute_instruction(instr.INSTR_Z, [0], physical=False)
                    self.node.keyout[self._othernode.name].append(([t,base],1)) # Ajout à la clé de chiffrement 
                
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


def Intermédiaire(Lalice, Lbob):
    """Un permet l'obstention d'un clé intermédiaire qui sera utilisée plus tard pour communiquer avec Lbob
     
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

class  TransmitProtocol(NodeProtocol):
    """Protocol performed by a Qonnector to transmit a qubit sent by a Qlient or a satellite to another Qlient
        
        Parameters
         Qlient_from: node from which a qubit is expected
         Qlient_to: node to which transmit the qubit received
         switch_succ: probability that the transmission succeeds"""
        
    def __init__(self, QNode_from, QNode_to, switch_succ, node=None, name=None):
                super().__init__(node=node, name=name)
                self._QNode_from = QNode_from
                self._QNode_to = QNode_to
                self._switch_succ=switch_succ
        
    def run(self):
            rec_mem = self.node.subcomponents["QNodeMemoryTo{}".format(self._QNode_from.name)]
            rec_port = self.node.ports[self.node.portsDict[self._QNode_from.name][1]]
            sen_mem = self.node.subcomponents["QNodeMemoryTo{}".format(self._QNode_to.name)]
            #print(sen_mem)
            while True:
                rec_mem.reset()
                sen_mem.reset()
                
                yield self.await_port_input(rec_port)
                #print("qubit received at qonnector" )
                t = self.node.ports["cin_{}".format(self._QNode_from.name)].rx_input()
                
                rec_mem.pop([0], skip_noise=True, meta_data={'internal': sen_mem})
                #print("qubit moved in qonnector's memory")               
                
                b = bernoulli.rvs(self._switch_succ)
                if b ==1 :
                    qubit, = sen_mem.pop([0])
                    self.node.ports["cout_{}".format(self._QNode_to.name)].tx_output(t)
                    #print("qubit sent to node")
                    
#Classical post-processing functions
def getTime(t):
    a,b = t
    return a[0]



def addDarkCounts(L,pdark, K):
    """Function to add dark counts to a list of outcomes. With probability pdark it will add an outcome to a 
     timestep where nothing was measured and with probability pdark/2 it will discard an outcome
     
    Parameters:
        L : List of outcomes
        pdark: probability of getting a dark count at a particular timestep
        K : Last timestep"""
    i = 0
    listestep = []
    for j in L:
        a, b = j
        listestep.append(a[0])
    while i< K:
        if i not in listestep:
            b = bernoulli.rvs(pdark)
            if b == 1:
                randbit = bernoulli.rvs(0.5)
                base = bernoulli.rvs(0.5)
                if base < 0.5:
                    L.append(([i,"plusmoins"],randbit))
                else:
                    L.append(([i,"zeroun"],randbit))
            i=i+1
        else:
            i=i+1
    L.sort(key=getTime)
    for e in L:
        if bernoulli.rvs(pdark/2)==1:
            L.remove(e)
            
def estimQBER(L):
    """Function to estimate the QBER from a list of couple (qubit sent,qubit measured)"""
    if L != []:
        Lres = []
        for i in L:
            (a,b)=i
            if a != b:
                Lres.append(b)
        return len(Lres)/len(L)
    else:
        return 1
