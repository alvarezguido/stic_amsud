#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 An updated version of LoRaSim 0.2.1 to simulate collisions in confirmable
 LoRaWAN with a datasize target
 author: Khaled Abdelfadeel khaled.abdelfadeel@mycit.ie
"""

"""
 SYNOPSIS:
   ./confirmablelorawan.py <nodes> <avgsend> <datasize> <full_collision> <randomseed>
 DESCRIPTION:
    nodes
        number of nodes to simulate
    avgsend
        average sending interval in seconds
    datasize
        Size of data that each device sends in bytes
    full_collision
        0   simplified check. Two packets collide when they arrive at the same time, on the same frequency and SF
        1   considers the capture effect
        2   consider the Non-orthognality SFs effect and capture effect
    randomseed
        random seed
 OUTPUT
    The result of every simulation run will be appended to a file named expX.dat,
    whereby X is the experiment number. The file contains a space separated table
    of values for nodes, collisions, transmissions and total energy spent. The
    data file can be easily plotted using e.g. gnuplot.
"""

import simpy
import random
import numpy as np
import math
import sys
import re
import matplotlib.pyplot as plt
import os
import operator

mode_debbug = 1 ##testing under IPython

if not mode_debbug:
    nrNodes = int(sys.argv[1])
    seed = int(sys.argv[2])
else:
    nrNodes = 10
    seed = 1


### VARIABLES ###
logs = []
avgSendTime = 1*60 ##This is 1/lambda, KIND of poisson distribution for transmissions.
datasize = 20000000
# packet size per SFs
#PcktLength_SF = [20,20,20,20,20,20]
PcktLength_SF = [228,228,228,228,228,228]
#Channels = [872000000, 864000000, 860000000]
Channels = [860000000]
full_collision = 0
Rnd = random.seed(seed)
simtime = 20*60 #in seconds
# maximum number of packets the BS can receive at the same time
maxBSReceives = 100

## outputs, prints, graphics stuff
show_placement = 0 #print the placement of the nodes.
graphics = 0 #turn on/off graphics
save_data = 0 #to save data to file
show_prints = 0
show_results = 1

old_stdout = sys.stdout
if not show_prints:
    null = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = null


# Bandwidth
Bandwidth = 125
# Coding Rate
CodingRate = 1 # Then, coding rate is 4/(CR + 4) --> 4/5 is default for LoRaWAN
LorawanHeader = 7
# last time the gateway acked a package
#nearstACK1p = [0,0,0] # 3 channels with 1% duty cycle
nearstACK1p = [0]*len(Channels) # N channels with 1% duty cycle
nearstACK10p = 0 # one channel with 10% duty cycle
AckMessLen = 0


gw_txpower = 14 #transmit power for gw
# this is an array with measured values for sensitivity
# see paper, Table 3
#sf7 = np.array([7,-126.5,-124.25,-120.75])
#sf8 = np.array([8,-127.25,-126.75,-124.0])
#sf9 = np.array([9,-131.25,-128.25,-127.5])
#sf10 = np.array([10,-132.75,-130.25,-128.75])
#sf11 = np.array([11,-134.5,-132.75,-130])
#sf12 = np.array([12,-133.25,-132.25,-132.25])
sf7 = np.array([7,-123,-120,-117.0])
sf8 = np.array([8,-126,-123,-120.0])
sf9 = np.array([9,-129,-126,-123.0])
sf10 = np.array([10,-132,-129,-126.0])
sf11 = np.array([11,-134.53,-131.52,-128.51])
sf12 = np.array([12,-137,-134,-131.0])

sensi = np.array([sf7,sf8,sf9,sf10,sf11,sf12])

IS7 = np.array([1,-8,-9,-9,-9,-9])
IS8 = np.array([-11,1,-11,-12,-13,-13])
IS9 = np.array([-15,-13,1,-13,-14,-15])
IS10 = np.array([-19,-18,-17,1,-17,-18])
IS11 = np.array([-22,-22,-21,-20,1,-20])
IS12 = np.array([-25,-25,-25,-24,-23,1])
IsoThresholds = np.array([IS7,IS8,IS9,IS10,IS11,IS12])



#
# packet error model assumming independent Bernoulli
#
from scipy.stats import norm
def ber_reynders(eb_no, sf):
    """Given the energy per bit to noise ratio (in db), compute the bit error for the SF"""
    return norm.sf(math.log(sf, 12)/math.sqrt(2)*eb_no)

def ber_reynders_snr(snr, sf, bw, cr):
    """Compute the bit error given the SNR (db) and SF"""
    Temp = [4.0/5,4.0/6,4.0/7,4.0/8]
    CR = Temp[cr-1]
    BW = bw*1000.0
    eb_no =  snr - 10*math.log10(BW/2**sf) - 10*math.log10(sf) - 10*math.log10(CR) + 10*math.log10(BW)
    return ber_reynders(eb_no, sf)

def per(sf,bw,cr,rssi,pl):
    snr = rssi  +174 - 10*math.log10(bw) - 6
    return False
    return 1 - (1 - ber_reynders_snr(snr, sf, bw, cr))**(pl*8)

#
# check for collisions at base station
# Note: called before a packet (or rather node) is inserted into the list
def checkcollision(packet):
    col = 0 # flag needed since there might be several collisions for packet
    processing = 0
    for i in range(0,len(packetsAtBS)):
        if packetsAtBS[i].packet.processed == 1:
            processing = processing + 1
    if (processing > maxBSReceives):
        print ("{:3.5f} || Too much packets on Base Station.. Packet is no Processed", len(packetsAtBS))
        packet.processed = 0
    else:
        packet.processed = 1

    if packetsAtBS:
        print ("{:3.5f} || CHECK node {} (sf:{} bw:{} freq:{:1.2e}) others: {}...".format(
             env.now,packet.nodeid, packet.sf, packet.bw, packet.freq, len(packetsAtBS)))
        for other in packetsAtBS:
            if other.nodeid != packet.nodeid:
               print ("{:3.5f} || ...others: node {} (sf:{} bw:{} freq:{:1.2e})...".format(
                   env.now, other.nodeid, other.packet.sf, other.packet.bw, other.packet.freq))
               if(full_collision == 1 or full_collision == 2):
                   if frequencyCollision(packet, other.packet) \
                   and timingCollision(packet, other.packet):
                       # check who collides in the power domain
                       if (full_collision == 1):
                          # Capture effect
                          c = powerCollision_1(packet, other.packet)
                       else:
                          # Capture + Non-orthognalitiy SFs effects
                          c = powerCollision_2(packet, other.packet)
                       # mark all the collided packets
                       # either this one, the other one, or both
                       for p in c:
                          p.collided = 1
                          if p == packet:
                            col = 1
                   else:
                       # no freq or timing collision, all fine
                       pass
               else:
                   # simple collision (full_colision=0)
                   if frequencyCollision(packet, other.packet) \
                   and sfCollision(packet, other.packet):
                       packet.collided = 1
                       other.packet.collided = 1  # other also got lost, if it wasn't lost already
                       col = 1
        return col
    return 0

# check if the gateway can ack this packet at any of the receive windows
# based on the duy cycle
def checkACK(packet):
    global  nearstACK1p
    global  nearstACK10p
    global Channels
    packet.acked =1
    return packet.acked
    # check ack in the first window
    chanlindex=Channels.index(packet.freq)
    timeofacking = env.now + 1  # one sec after receiving the packet
    print ("{:3.5f} || sending 1st ack from gw to node {}".format(timeofacking,packet.nodeid))
    if (timeofacking >= nearstACK1p[chanlindex]):
        # this packet can be acked
        packet.acked = 1
        tempairtime = airtime(packet.sf, CodingRate, AckMessLen+LorawanHeader, Bandwidth)
        nearstACK1p[chanlindex] = timeofacking+(tempairtime/0.01) #Why?
        nodes[packet.nodeid].rxtime += tempairtime
        return packet.acked ##exit the function, no needed for 2nd ACK
    else:
        # this packet can not be acked
        packet.acked = 0
        Tsym = (2**packet.sf)/(Bandwidth*1000) # sec
        Tpream = (8 + 4.25)*Tsym
        nodes[packet.nodeid].rxtime += Tpream

    # check ack in the second window
    timeofacking = env.now + 2  # two secs after receiving the packet
    print ("{:3.5f} || SENDING 2nd ack from gw to node {}".format(timeofacking,packet.nodeid))
    if (timeofacking >= nearstACK10p):
        # this packet can be acked
        packet.acked = 1
        tempairtime = airtime(12, CodingRate, AckMessLen+LorawanHeader, Bandwidth)
        nearstACK10p = timeofacking+(tempairtime/0.1)
        nodes[packet.nodeid].rxtime += tempairtime
        return packet.acked
    else:
        # this packet can not be acked
        print ("{:3.5f} || This packet cant be acked for node {} ".format(timeofacking,packet.nodeid))
        packet.acked = 0
        Tsym = (2.0**12)/(Bandwidth*1000.0) # sec
        Tpream = (8 + 4.25)*Tsym
        nodes[packet.nodeid].rxtime += Tpream
        return packet.acked

#
# frequencyCollision, conditions
#
#        |f1-f2| <= 120 kHz if f1 or f2 has bw 500
#        |f1-f2| <= 60 kHz if f1 or f2 has bw 250
#        |f1-f2| <= 30 kHz if f1 or f2 has bw 125
def frequencyCollision(p1,p2):
    if (abs(p1.freq-p2.freq)<=120 and (p1.bw==500 or p2.freq==500)):
        print ("{:3.5f} || ...Freq coll at bw 500 between node {} and node {}.. Let's check SF.."
               .format(env.now,p1.nodeid,p2.nodeid))
        return True
    elif (abs(p1.freq-p2.freq)<=60 and (p1.bw==250 or p2.freq==250)):
        print ("{:3.5f} || ...Freq coll at bw 250 between node {} and node {}.. Let's check SF.."
               .format(env.now,p1.nodeid,p2.nodeid))
        return True
    else:
        if (abs(p1.freq-p2.freq)<=30):
            print ("{:3.5f} || ...Freq coll at bw 125 between node {} and node {}.. Let's check SF.."
                   .format(env.now,p1.nodeid,p2.nodeid))
            return True
        #else:
    print ("{:3.5f} || ...No freq coll between node {} and node {}.. Let's check SF.."
           .format(env.now,p1.nodeid,p2.nodeid))
    return False

def sfCollision(p1, p2):
    global full_collision
    if p1.sf == p2.sf:
        if full_collision == 0:
            print ("{:3.5f} || SF coll on sf {}, between node {} and node {}.. Packet collided!!"
                   .format(env.now,p1.sf,p1.nodeid, p2.nodeid))
            # p2 may have been lost too, will be marked by other checks
            return True
        else:
             print ("{:3.5f} || SF coll on sf {}, between node {} and node {}.. Let's check capture effect.."
                   .format(env.now,p1.sf,p1.nodeid, p2.nodeid))
             # p2 may have been lost too, will be marked by other checks
             return True
    print ("{:3.5f} || No SF collision between node {} and node {}"
           .format(env.now,p1.nodeid,p2.nodeid))
    return False

# check only the capture between the same spreading factor
def powerCollision_1(p1, p2):
    #powerThreshold = 6
    print ("{:3.5f} || CHECK power: node {} with {:3.2f} dBm and node {} with {:3.2f} dBm; difference is {:3.2f} dBm"
           .format(env.now,p1.nodeid,p1.rssi, p2.nodeid,p2.rssi, round(p1.rssi - p2.rssi,2)))
    if p1.sf == p2.sf:
       if abs(p1.rssi - p2.rssi) < IsoThresholds[p1.sf-7][p2.sf-7]:
            print ("{:3.5f} || Power Collision: both node {} and node {} are collided!!"
                   .format(env.now,p1.nodeid, p2.nodeid))
            # packets are too close to each other, both collide
            # return both pack ets as casualties
            return (p1, p2)
       elif p1.rssi - p2.rssi < IsoThresholds[p1.sf-7][p2.sf-7]:
            # p2 overpowered p1, return p1 as casualty
            print ("{:3.5f} || Power coll: node {} overpowered node {}; only last is collided!!"
                   .format(env.now, p2.nodeid, p1.nodeid))
            return (p1,)
            print ("{:3.5f} || Power coll: node {} overpowered node {}; only last is collided!!"
                   .format(env.now, p1.nodeid, p2.nodeid))
       # p2 was the weaker packet, return it as a casualty
       return (p2,)
    else:
       return ()

# check the capture effect and checking the effect of pesudo-orthognal SFs
def powerCollision_2(p1, p2):
    #powerThreshold = 6
    print ("{:3.5f} || CHECK SF: node {} with sf {} and node {} with sf {}"
           .format(env.now,p1.nodeid,p1.sf, p2.nodeid, p2.sf))
    print ("{:3.5f} || CHECK power: node {} with {:3.2f} dBm and node {} with {:3.2f} dBm; difference is {:3.2f} dBm"
           .format(env.now,p1.nodeid,p1.rssi, p2.nodeid,p2.rssi, round(p1.rssi - p2.rssi,2)))
    if p1.sf == p2.sf:
       if abs(p1.rssi - p2.rssi) < IsoThresholds[p1.sf-7][p2.sf-7]:
           print ("{:3.5f} || Power Collision for both node {} and node {}"
                  .format(env.now,p1.nodeid, p2.nodeid))
           # packets are too close to each other, both collide
           # return both packets as casualties
           return (p1, p2)
       elif p1.rssi - p2.rssi < IsoThresholds[p1.sf-7][p2.sf-7]:
           # p2 overpowered p1, return p1 as casualty
           print ("{:3.5f} || Power coll: node {} overpowered node {}; only last is collided!!"
                   .format(env.now, p2.nodeid, p1.nodeid))
           return (p1,)
       print ("{:3.5f} || Power coll: node {} overpowered node {}; only last is collided!!"
                   .format(env.now, p1.nodeid, p2.nodeid))
       # p2 was the weaker packet, return it as a casualty
       return (p2,)
    else:
       if p1.rssi-p2.rssi > IsoThresholds[p1.sf-7][p2.sf-7]:
          #print ("{:3.5f} || Power")
          if p2.rssi-p1.rssi > IsoThresholds[p2.sf-7][p1.sf-7]:
              #print ("p2 is OK")
              return ()
          else:
              print ("{:3.5f} || Power is not enough for node {}, then is collided!!"
                     .format(env.now,p2.nodeid))
              return (p2,)
       else:
           print ("{:3.5f} || Power is not enough for node {}, then is collided!!"
                  .format(env.now,p1.nodeid))
           if p2.rssi-p1.rssi > IsoThresholds[p2.sf-7][p1.sf-7]:
               #print ("p2 is OK")
               return (p1,)
           else:
               print ("{:3.5f} || Power is not enough for node {}, then is collided!!"
                     .format(env.now,p2.nodeid))
               return (p1,p2)


def timingCollision(p1, p2):
    # assuming p1 is the freshly arrived packet and this is the last check
    # we've already determined that p1 is a weak packet, so the only
    # way we can win is by being late enough (only the first n - 5 preamble symbols overlap)

    # assuming 8 preamble symbols
    Npream = 8 ##default value

    # we can lose at most (Npream - 5) * Tsym of our preamble
    Tpreamb = 2**p1.sf/(1.0*p1.bw) * (Npream - 5)

    # check whether p2 ends in p1's critical section
    p2_end = p2.addTime + p2.rectime
    p1_cs = env.now + (Tpreamb/1000.0)  # to sec
    print ("{:3.5f} || Timing coll for node {} ({},{},{}) and node {} ({},{})"
           .format(env.now,p1.nodeid, env.now - env.now, p1_cs - env.now, p1.rectime,
           p2.nodeid, p2.addTime - env.now, p2_end - env.now
    ))
    if p1_cs < p2_end:
        # p1 collided with p2 and lost
        print ("{:3.5f} || Not late enough.. Let's check power"
               .format(env.now))
        return True
    print ("{:3.5f} || Late enough... Saved by the preamble"
           .format(env.now))
    return False


# this function computes the airtime of a packet
# according to LoraDesignGuide_STD.pdf

#Check out https://loratools.nl/#/airtime

def airtime(sf,cr,pl,bw):
    H = 0        # implicit header disabled (H=0) or not (H=1)
    DE = 0       # low data rate optimization enabled (=1) or not (=0)
    Npream = 8   # number of preamble symbol (12.25  from Utz paper)

    if bw == 125 and sf in [11, 12]:
        # low data rate optimization mandated for BW125 with SF11 and SF12
        DE = 1
    if sf == 6:
        # can only have implicit header with SF6
        H = 1

    Tsym = (2.0**sf)/bw  # msec
    Tpream = (Npream + 4.25)*Tsym
    #print (">> Has sf {}, cr {}, pl {}, bw {}".format(sf,cr,pl,bw))
    payloadSymbNB = 8 + max(math.ceil((8.0*pl-4.0*sf+28+16-20*H)/(4.0*(sf-2*DE)))*(cr+4),0)
    Tpayload = payloadSymbNB * Tsym
    return ((Tpream + Tpayload)/1000.0)  # to secs
    #return 0.368896 #hardcoded
#
# this function creates a node
#
class myNode():
    def __init__(self, nodeid, bs, period, datasize):
        self.nodeid = nodeid
        self.buffer = datasize
        self.bs = bs
        self.first = 1
        self.period = period
        self.lstretans = 0
        self.sent = 0
        self.coll = 0
        self.lost = 0
        self.noack = 0
        self.acklost = 0
        self.recv = 0
        self.losterror = 0
        self.rxtime = 0
        self.x = 0
        self.y = 0

        # this is very complex prodecure for placing nodes
        # and ensure minimum distance between each pair of nodes
        found = 0
        rounds = 0
        global nodes
        while (found == 0 and rounds < 100):
            a = random.random()
            b = random.random()
            if b<a:
                a,b = b,a
            posx = b*maxDist*math.cos(2*math.pi*a/b)+bsx
            posy = b*maxDist*math.sin(2*math.pi*a/b)+bsy
            if len(nodes) > 0:
                for index, n in enumerate(nodes):
                    dist = np.sqrt(((abs(n.x-posx))**2)+((abs(n.y-posy))**2))
                    if dist >= 10:
                        found = 1
                        self.x = posx
                        self.y = posy
                    else:
                        rounds = rounds + 1
                        if rounds == 1000:
                            print ("could not place new node, giving up")
                            exit(-1)
            else:
                #print ("{:3.5f} || This is first node..".format(env.now))
                self.x = posx
                self.y = posy
                found = 1
        self.dist = np.sqrt((self.x-bsx)*(self.x-bsx)+(self.y-bsy)*(self.y-bsy))
        if show_placement == 1:
            print('{:3.5f} || Placement: Node {}, Pos X {:4.2f}, Pos Y {:4.2f}, dist to gw: {:4.2f}'
                  .format(env.now,nodeid,self.x,self.y, self.dist))

        self.txpow = 0

        # graphics for node
        global graphics
        if (graphics == 1):
            global ax
            ax.add_artist(plt.Circle((self.x, self.y), 0.25, fill=True, color='blue'))


class assignParameters():
    def __init__(self, nodeid, distance):
        global Ptx
        global gamma
        global d0
        global var
        global Lpld0
        global GL
        global show_placement

        self.nodeid = nodeid
        self.txpow = 14
        self.bw = Bandwidth
        self.cr = CodingRate
        self.sf = 12
        self.rectime = airtime(self.sf, self.cr, LorawanHeader+PcktLength_SF[self.sf-7], self.bw)
        #self.freq = random.choice([872000000, 864000000, 860000000])
        self.freq = random.choice(Channels)

        Prx = self.txpow  ## zero path loss by default
        # log-shadow
        Lpl = Lpld0 + 10*gamma*math.log10(distance/d0) + var
        #print ("Lpl:", Lpl)
        Prx = self.txpow - GL - Lpl
        minairtime = 9999
        minsf = 0
        minbw = 0
        #print ("Prx:", Prx)
        #### CHOOSE SF ACCORDING TO DISTANCE ####
        for i in range(0,6):  # SFs
            if ((sensi[i, [125,250,500].index(self.bw) + 1]) < Prx):
                at = airtime(i+7, self.cr, LorawanHeader+PcktLength_SF[i], self.bw)
                if at < minairtime:
                    minairtime = at
                    minsf = i+7
                    minsensi = sensi[i, [125,250,500].index(self.bw) + 1]
        if show_placement == 1:
            print ("{:3.5f} || Placement: Node {} best sf {}, best ToA: {:2.4f},bw {}Hz"
                   .format(env.now,nodeid,minsf,minairtime,self.bw))
            #print ("best sf:", minsf, " best bw: ", minbw, "best airtime:", minairtime)
        if (minsf != 0):
            self.rectime = minairtime
            self.sf = minsf

        # SF, BW, CR and PWR distributions
        #print ("bw", self.bw, "sf", self.sf, "cr", self.cr)
        global SFdistribution, CRdistribution, TXdistribution, BWdistribution,FRdistribution
        SFdistribution[self.sf-7]+=1;
        CRdistribution[self.cr-1]+=1;
        TXdistribution[int(self.txpow)-2]+=1;
        FRdistribution[Channels.index(self.freq)]+=1;
        if self.bw==125:
            BWdistribution[0]+=1;
        elif self.bw==250:
            BWdistribution[1]+=1;
        else:
            BWdistribution[2]+=1;

#
# this function creates a packet (associated with a node)
# it also sets all parameters, currently random
#
class myPacket():
    def __init__(self, nodeid, freq, sf, bw, cr, txpow, distance):
        global gamma
        global d0
        global var
        global Lpld0
        global GL

        self.nodeid = nodeid
        self.freq = freq
        self.sf = sf
        self.bw = bw
        self.cr = cr
        self.txpow = txpow
        # transmission range, needs update XXX
        self.transRange = 150
        self.pl = LorawanHeader+PcktLength_SF[self.sf-7]
        self.symTime = (2.0**self.sf)/self.bw
        self.arriveTime = 0
        if var == 0:
            Lpl = Lpld0 + 10*gamma*math.log10(distance/d0)
        else:
            Lpl = Lpld0 + 10*gamma*math.log10(distance/d0) + np.random.normal(-var, var)

        self.rssi = self.txpow - GL - Lpl
        self.rectime = airtime(self.sf,self.cr,self.pl,self.bw)
        if show_placement == 1:
            print ("{:3.5f} || Placement: Node {}, symTime {}, rssi {:3.3f}, rectime {:3.3f}"
                   .format(env.now,self.nodeid,self.symTime, self.rssi,self.rectime))
         
        # denote if packet is collided
        self.collided = 0
        self.processed = 0
        self.lost = False
        self.perror = False
        self.acked = 0
        self.acklost = 0

#
# main discrete event loop, runs for each node
# a global list of packet being processed at the gateway
# is maintained
#
def transmit(env,node):
    while node.buffer > 0.0:
        node.packet.rssi = node.packet.txpow - Lpld0 - 10*gamma*math.log10(node.dist/d0) - np.random.normal(-var, var)
        # add maximum number of retransmissions
# =============================================================================
#         if (node.lstretans and node.lstretans <= 8):
#             node.first = 0
#             node.buffer += PcktLength_SF[node.parameters.sf-7]
#             # the randomization part (2 secs) to resove the collisions among retrasmissions
#             yield env.timeout(max(2+airtime(12, CodingRate, AckMessLen+LorawanHeader, Bandwidth), float(node.packet.rectime*((1-0.01)/0.01)))+(random.expovariate(1.0/float(2000))/1000.0))
#             print ("{:3.5f} || sending re-tx from node {} to gw".format(env.now,node.nodeid))
# =============================================================================
        
        node.first = 0
        node.lstretans = 0
        ##poisson distribution until send next packet
        yield env.timeout(random.expovariate(1.0/float(node.period)))
        #yield env.timeout(60)
        print ("{:3.5f} || sending tx from node {} to gw".format(env.now,node.nodeid))
           


        node.buffer -= PcktLength_SF[node.parameters.sf-7]
# =============================================================================
#         print ("{:3.5f} || Node {}, has a buffer of {} bytes"
#                .format(env.now,node.nodeid,node.buffer))
# =============================================================================

        # time sending and receiving
        # packet arrives -> add to base station
        node.sent = node.sent + 1
        if (node in packetsAtBS):
            print ("ERROR: packet already in")
        else:
            sensitivity = sensi[node.packet.sf - 7, [125,250,500].index(node.packet.bw) + 1]
            if node.packet.rssi < sensitivity:
                print ("{:3.5f} || node {}: packet will be lost due Lpl"
                       .format(env.now,node.nodeid))
                node.packet.lost = True
            else:
                node.packet.lost = False
                if (per(node.packet.sf,node.packet.bw,node.packet.cr,node.packet.rssi,node.packet.pl) < random.uniform(0,1)):
                    # OK CRC
                    node.packet.perror = False
                else:
                    # Bad CRC
                    node.packet.perror = True
                # adding packet if no collision
                if (checkcollision(node.packet)==1):
                    node.packet.collided = 1
                else:
                    node.packet.collided = 0

                packetsAtBS.append(node)
                node.packet.addTime = env.now
                
        ###time on air of the packet
        yield env.timeout(node.packet.rectime)
        

        if (node.packet.lost == 0\
                and node.packet.perror == False\
                and node.packet.collided == False\
                and checkACK(node.packet)):
            node.packet.acked = 1
            # the packet can be acked
            # check if the ack is lost or not
            global gw_txpower
            if((gw_txpower - Lpld0 - 10*gamma*math.log10(node.dist/d0) - np.random.normal(-var, var)) > sensi[node.packet.sf-7, [125,250,500].index(node.packet.bw) + 1]):
            # the ack is not lost
                node.packet.acklost = 0
            else:
            # ack is lost
                node.packet.acklost = 1
        else:
            node.packet.acked = 0

        if node.packet.processed == 1:
            global nrProcessed
            nrProcessed = nrProcessed + 1
        if node.packet.lost:
            #node.buffer += PcktLength_SF[node.parameters.sf-7]
# =============================================================================
#             print ("{:3.5f} || Node {}, has a buffer of {} bytes"
#                .format(env.now,node.nodeid,node.buffer))
# =============================================================================
            node.lost = node.lost + 1
            node.lstretans += 1
            global nrLost
            nrLost += 1
        elif node.packet.perror:
# =============================================================================
#             print ("{:3.5f} || Node {}, has a buffer of {} bytes"
#                .format(env.now,node.nodeid,node.buffer))
# =============================================================================
            node.losterror = node.losterror + 1
            global nrLostError
            nrLostError += 1
        elif node.packet.collided == 1:
            #node.buffer += PcktLength_SF[node.parameters.sf-7]
# =============================================================================
#             print ("{:3.5f} || Node {}, has a buffer of {} bytes"
#                .format(env.now,node.nodeid,node.buffer))
# =============================================================================
            node.coll = node.coll + 1
            node.lstretans += 1
            global nrCollisions
            nrCollisions = nrCollisions +1
        elif node.packet.acked == 0:
            #node.buffer += PcktLength_SF[node.parameters.sf-7]
# =============================================================================
#             print ("{:3.5f} || Node {}, has a buffer of {} bytes"
#                .format(env.now,node.nodeid,node.buffer))
# =============================================================================
            node.noack = node.noack + 1
            node.lstretans += 1
            global nrNoACK
            nrNoACK += 1
        elif node.packet.acklost == 1:
            #node.buffer += PcktLength_SF[node.parameters.sf-7]
# =============================================================================
#             print ("{:3.5f} || Node {}, has a buffer of {} bytes"
#                .format(env.now,node.nodeid,node.buffer))
# =============================================================================
            node.acklost = node.acklost + 1
            node.lstretans += 1
            global nrACKLost
            nrACKLost += 1
        else:
            node.recv = node.recv + 1
            node.lstretans = 0
            global nrReceived
            nrReceived = nrReceived + 1

        # complete packet has been received by base station
        # can remove it
        if (node in packetsAtBS):
            packetsAtBS.remove(node)
         
        if node.packet.lost:
            logs.append("{:3.3f},{},{:3.3f},{:3.3f},{},PL".format(env.now,node.nodeid,node.dist,0,node.packet.sf))
        elif node.packet.collided:
            logs.append("{:3.3f},{},{:3.3f},{:3.3f},{},PC".format(env.now,node.nodeid,node.dist,0,node.packet.sf))
        elif node.packet.processed == 0:
            logs.append("{:3.3f},{},{:3.3f},{:3.3f},{},NP".format(env.now,node.nodeid,node.dist,0,node.packet.sf))
        else:
            logs.append("{:3.3f},{},{:3.3f},{:3.3f},{},PE".format(env.now,node.nodeid,node.dist,0,node.packet.sf))
            # reset the packet
             
        node.packet.collided = 0
        node.packet.processed = 0
        node.packet.lost = False
        node.packet.acked = 0
        node.packet.acklost = 0

#
# "main" program
#

# global stuff
nodes = []
nodeder1 = [0 for i in range(0,nrNodes)]
nodeder2 = [0 for i in range(0,nrNodes)]
tempdists = [0 for i in range(0,nrNodes)]
packetsAtBS = []
SFdistribution = [0 for x in range(0,6)]
BWdistribution = [0 for x in range(0,3)]
CRdistribution = [0 for x in range(0,4)]
TXdistribution = [0 for x in range(0,13)]
FRdistribution = [0 for x in range(0,len(Channels))]
env = simpy.Environment()



# max distance: 300m in city, 3000 m outside (5 km Utz experiment)
# also more unit-disc like according to Utz
bsId = 1
nrCollisions = 0
nrReceived = 0
nrProcessed = 0
nrLost = 0
nrLostError = 0
nrNoACK = 0
nrACKLost = 0

Ptx = 9.75
gamma = 2.08
d0 = 40.0
var = 2.0
Lpld0 = 127.41
GL = 0 #antenna gain for nodes.
minsensi = np.amin(sensi[:,[125,250,500].index(Bandwidth) + 1])
Lpl = Ptx - minsensi
maxDist = d0*(10**((Lpl-Lpld0)/(10.0*gamma)))
maxDist = 10
print ("maxDist:", maxDist)

# base station placement
bsx = maxDist+10
bsy = maxDist+10
xmax = bsx + maxDist + 10
ymax = bsy + maxDist + 10

# prepare graphics and add sink
if (graphics == 1):
    plt.ion()
    plt.figure()
    ax = plt.gcf().gca()
    # XXX should be base station position
    ax.add_artist(plt.Circle((bsx, bsy), 0.5, fill=True, color='green'))
    ax.add_artist(plt.Circle((bsx, bsy), maxDist, fill=False, color='red'))

for i in range(0,nrNodes):
    # myNode takes period (in ms), base station id packetlen (in Bytes)
    node = myNode(i,bsId, avgSendTime, datasize)
    nodes.append(node)
    node.parameters = assignParameters(node.nodeid, node.dist)
    node.packet = myPacket(node.nodeid, node.parameters.freq,
                           node.parameters.sf, node.parameters.bw,
                           node.parameters.cr, node.parameters.txpow, node.dist)
    env.process(transmit(env,node))


#prepare show
if (graphics == 1):
    plt.xlim([0, xmax])
    plt.ylim([0, ymax])
    plt.draw()
    plt.show()

# start simulation
#env.run()
env.run(until=simtime)

# print stats and save into file
#print "nrCollisions ", nrCollisions

# compute energy
# Transmit consumption in mA from -2 to +17 dBm
TX = [22, 22, 22, 23,                                      # RFO/PA0: -2..1
      24, 24, 24, 25, 25, 25, 25, 26, 31, 32, 34, 35, 44,  # PA_BOOST/PA1: 2..14
      82, 85, 90,                                          # PA_BOOST/PA1: 15..17
      105, 115, 125]                                       # PA_BOOST/PA1+PA2: 18..20
RX = 16
V = 3.0     # voltage XXX
sent = sum(n.sent for n in nodes)
energy = sum(((node.packet.rectime * node.sent * TX[int(node.packet.txpow)+2])+(node.rxtime * RX)) * V  for node in nodes)  / 1e3

if show_results:
    sys.stdout = old_stdout 
    print ("\n============================")
    print("    *** PARAMETERS ***")
    print ("============================")
    print ("Nodes:", nrNodes)
    print ("DataSize [bytes]", datasize)
    print ("AvgSendTime (exp. distributed):",avgSendTime)
    print ("Full Collision: ", full_collision)
    print ("Random Seed: ", int(seed))

if show_results:
    sys.stdout = old_stdout
    print ("============================")
    print("    *** STATISTICS ***")
    print ("============================")
    print ("energy (in J): ", energy)
    print ("sent packets: ", sent)
    print ("collisions: ", nrCollisions)
    print ("received packets: ", nrReceived)
    print ("processed packets by BS: ", nrProcessed)
    print ("lost packets: ", nrLost)
    print ("lost packets due bad CRC: ", nrLostError)
    print ("NoACK packets (packets than can't be ACK by gw'): ", nrNoACK)
    print ("ACK lost packets (from gw to node): ",nrACKLost)
    toa = nodes[0].packet.rectime
    # data extraction rate
    der1 = (sent-nrCollisions)/float(sent) if sent!=0 else 0
    print ("DER (data extraction rate):", der1)
    der2 = (nrReceived)/float(sent) if sent!=0 else 0
    print ("DER (data extraction rate) method 2:", der2)
    S_sim = ((sent-nrCollisions)*toa)/simtime #both tx and re-tx
    print ("Throughtput S sim:",S_sim)
    print ("============================")
    print("   ANALYTICAL STATISTICS ")
    print ("============================")
    
# =============================================================================
#     p = (1/avgSendTime)*nodes[0].packet.rectime
#     print ("p:",p)
# =============================================================================
    
# =============================================================================
#     der2_analytic_1 = (1-p)**(2*(nrNodes-1))
#     print ("DER2 Analytic method 1:",der2_analytic_1)
# =============================================================================
    
    G = nrNodes*(1/avgSendTime)*nodes[0].packet.rectime
    print ("G, Offered Traffic generated (there's no re-tx):",G,"[erlangs]")
   
    #der2_analytic_2 = np.exp(-2*G)
    der2_analytic = np.exp(-2*((nrNodes-1)*(1/avgSendTime)*toa))
    der2_analytic = np.exp(-2*G)
    print ("DER Analytic:",der2_analytic)
   
# =============================================================================
#     S_analytic1 = nrNodes*p*der2_analytic_1
#     print ("Throughtput S analytic1:",S_analytic1)
# =============================================================================
    
    S_analytic2 = G*np.exp(-2*G) #considering re-transmisions
    print ("Throughtput S analytic:",S_analytic2)
    print ("---")
    print ("Throughtput S analytic:",S_analytic2)
    print ("Throughtput S sim:",S_sim)
    print ("DER Analytic:",der2_analytic)
    print ("DER sim:", der1) #considering also re-tx
    #print ("DER2 Analytic method 1:",der2_analytic_1)
    
    
 
# data extraction rate per node
for i in range(0,nrNodes):
    tempdists[i] = nodes[i].dist
    nodeder1[i] = ((nodes[i].sent-nodes[i].coll)/(float(nodes[i].sent)) if float(nodes[i].sent)!=0 else 0)
    nodeder2[i] = (nodes[i].recv/(float(nodes[i].sent)) if float(nodes[i].sent)!=0 else 0)
# calculate the fairness indexes per node
nodefair1 = (sum(nodeder1)**2/(nrNodes*sum([i*float(j) for i,j in zip(nodeder1,nodeder1)])) if (sum([i*float(j) for i,j in zip(nodeder1,nodeder1)]))!=0 else 0)
nodefair2 = (sum(nodeder2)**2/(nrNodes*sum([i*float(j) for i,j in zip(nodeder2,nodeder2)])) if (sum([i*float(j) for i,j in zip(nodeder2,nodeder2)]))!=0 else 0)

if show_results:
    sys.stdout = old_stdout
    print ("============================")
    print ("SFdistribution: ", SFdistribution)
    print ("BWdistribution: ", BWdistribution)
    print ("CRdistribution: ", CRdistribution)
    print ("TXdistribution: ", TXdistribution)
    print ("FRdistribution: ", FRdistribution)
    print ("CollectionTime: ", env.now)

# save experiment data into a dat file that can be read by e.g. gnuplot
# name of file would be:  exp0.dat for experiment 0
if save_data == 1:
    fname = str("confirmablelorawan") + ".dat"
    print (fname)
    if os.path.isfile(fname):
         res= "\n" + str(seed) + ", " + str(full_collision) + ", " + str(nrNodes) + ", " + str(avgSendTime) + ", " + str(datasize) + ", " + str(sent) + ", "  + str(nrCollisions) + ", "  + str(nrLost) + ", "  + str(nrLostError) + ", " +str(nrNoACK) + ", " +str(nrACKLost) + ", " + str(env.now)+ ", " + str(der1) + ", " + str(der2)  + ", " + str(energy) + ", "  + str(nodefair1) + ", "  + str(nodefair2) + ", "  + str(SFdistribution)
    else:
         res = "#randomseed, collType, nrNodes, TransRate, DataSize, nrTransmissions, nrCollisions, nrlost, nrlosterror, nrnoack, nracklost, CollectionTime, DER1, DER2, OverallEnergy, nodefair1, nodefair2, sfdistribution\n" + str(seed) + ", " + str(full_collision) + ", " + str(nrNodes) + ", " + str(avgSendTime) + ", " + str(datasize) + ", " + str(sent) + ", "  + str(nrCollisions) + ", "  + str(nrLost) + ", "  + str(nrLostError) + ", " +str(nrNoACK) + ", " +str(nrACKLost) + ", " + str(env.now)+ ", " + str(der1) + ", " + str(der2)  + ", " + str(energy) + ", "  + str(nodefair1) + ", "  + str(nodefair2) + ", "  + str(SFdistribution)
    newres=re.sub('[^#a-zA-Z0-9 \n\.]','',res)
    print (newres)
    with open(fname, "a") as myfile:
        myfile.write(newres)
    myfile.close()


if save_data:
    name = "lorawan_pAloha_noACK"
    RANDOM_SEED = seed
    folder = name+'_1CH_s'+str(RANDOM_SEED)
    if not os.path.exists(folder):
        os.makedirs(folder)
    fname = "./"+folder+"/" + str(name+"_1CH_"+"_s"+str(RANDOM_SEED))+".csv"
    if not os.path.exists(fname):
        with open(fname,"a") as myfile:
            logs = "0,{},0,0,0,0".format(seed)
            myfile.write(logs+"\n")
        myfile.close()
        
    with open(fname,"a") as myfile:
        logs = "{},{},{},{},{},{}".format(nrNodes,seed,der2_analytic,der1,S_analytic2,S_sim)
        myfile.write(logs+"\n")
    myfile.close()

# with open('nodes.txt','w') as nfile:
#     for n in nodes:
#         nfile.write("{} {} {}\n".format(n.x, n.y, n.nodeid))
# with open('basestation.txt', 'w') as bfile:
#     bfile.write("{} {} {}\n".format(bsx, bsy, 0)
