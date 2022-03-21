#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 07:49:24 2022

@author: guido
"""
import matplotlib.pyplot as plt

seed = 1
file_name="lorawan_pAloha_noACK_3CH_s1/lorawan_pAloha_noACK_3CH__s{}.csv".format(seed)
with open(file_name,"r") as reader: ### SELECT FILE TO WORK WITH
    lista = reader.read().splitlines()

nodes = []
der_a = []
der_sim = []
s_a = []
s_sim = []
for line in lista:
    datos = line.split(",")
    nodes.append(int(datos[0]))
    der_a.append(float(datos[2]))
    der_sim.append(float(datos[3]))
    s_a.append(float(datos[4]))
    s_sim.append(float(datos[5]))


    
plt.figure(figsize=(8, 6), dpi=80)
plt.grid()
font_size= 12
#plt.rcParams.update({'font.size': 8})


plt.plot(nodes, der_a,'o-', color="lightcoral" ,label = "DER analitico")
plt.xlabel('N° de nodos', fontsize=font_size)
#plt.ylim(0,1.1)
#plt.xlim(0,110)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.ylabel('DER',fontsize=font_size)
plt.title("Data extraction Rate",fontsize=font_size)

plt.plot(nodes, der_sim,'o-', color="royalblue" ,label = "DER simulado")
plt.legend(bbox_to_anchor=(0.95,0.5), loc="center left", borderaxespad=-4, fontsize=font_size-2)

plt.savefig("plots/metrics-DER-3CH-s{}.png".format(seed), format='png', dpi=300)
plt.savefig("plots/metrics-DER-3CH-s{}.pdf".format(seed), format='pdf')


###throughput plots
plt.figure(figsize=(8, 6), dpi=80)
plt.grid()
font_size= 12
#plt.rcParams.update({'font.size': 8})

plt.plot(nodes, s_a,'o-', color="lightcoral" ,label = "S analitico")
plt.xlabel('N° de nodos', fontsize=font_size)
#plt.ylim(0,1.1)
#plt.xlim(0,110)
plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)
plt.ylabel('S',fontsize=font_size)
plt.title("Throughput S",fontsize=font_size)

plt.plot(nodes, s_sim,'o-', color="royalblue" ,label = "S simulado")
plt.legend(bbox_to_anchor=(0.95,0.5), loc="center left", borderaxespad=-4, fontsize=font_size-2)

plt.savefig("plots/metrics-S-3CH-s{}.png".format(seed), format='png', dpi=300)
plt.savefig("plots/metrics-S-3CH-s{}.pdf".format(seed), format='pdf')
