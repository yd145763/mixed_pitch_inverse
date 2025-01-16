# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 18:01:54 2025

@author: Lim Yudian
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:27:35 2025

@author: Lim Yudian
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import imp
from matplotlib.ticker import StrMethodFormatter
import re

small_pitch_list = []
big_pitch_list = []
small_count_list = []
big_count_list = []
small_dc_list = []
big_dc_list = []
E20_list = []
E40_list = []
E60_list = []
E80_list = []


full_power_distribution = False
visualize_index = False


wg_length = 10e-6
offset = 5e-6

# Specify the path
path = 'C:\\Users\\Lim Yudian\\Documents\\mixed_pitch_gratings\\'

# Get a list of all files in the directory
files = os.listdir(path)

# Filter the GDS files and extract their names without the extension
gds_files = [filename[:-4] for filename in files if filename.endswith('end.gds')]
gds_files.sort(key=str.lower)



ran_gds_files = pd.read_csv(path+'ranlist.csv', index_col = 0)
ran_gds_files = ran_gds_files['ran_gds']
ran_gds_files = [i for i in ran_gds_files]

numbers = np.arange(4000, len(gds_files), 2000)
numbers = [i for i in numbers]
numbers.append(len(gds_files))

for n in numbers:

    for g in gds_files[:n]:
        if g in ran_gds_files:
            print('data already acquired')
        if g not in ran_gds_files:
            os.add_dll_directory("C:\\Program Files\\Lumerical\\v241\\api\\python\\lumapi.py") #the lumapi.py path in your pc, remember the double \\
            lumapi = imp.load_source("lumapi","C:\\Program Files\\Lumerical\\v241\\api\\python\\lumapi.py") #the lumapi.py path in your pc, remember the double \\
            fdtd = lumapi.FDTD('C:\\Users\\Lim Yudian\\Documents\\mixed_pitch_gratings\\script_run.fsp')
               
            print(g)
            fdtd.switchtolayout()
            fdtd.deleteall()
            
            def extractfromstring(symbol, filename):
                match = re.search(symbol+'(\\d+)_', filename)
                if match:
                    number = match.group(1)
                    number = int(number)
                    in_micron = number/1000
                    for_fdtd = in_micron*1e-6
                    return for_fdtd
                
            def extractfromstringinteger(symbol, filename):
                match = re.search(symbol+'(\\d+)_', filename)
                if match:
                    number = match.group(1)
                    number = int(number)
                    for_fdtd = number
                    return for_fdtd
            
            small_pitch = extractfromstring('sp', g)
            
            big_pitch = extractfromstring('bp', g)        
            
            small_count = extractfromstringinteger('sc', g)
            
            
            big_count = extractfromstringinteger('bc', g)
            
            
            small_dc = extractfromstring('sdc', g)
            
            
            big_dc = extractfromstring('bdc', g)
            
            
    
            
            total_device_length = wg_length+(big_count*big_pitch)+(small_count*small_pitch)+(0.5*small_pitch)+offset
            
            
            fdtd.addfdtd()
            fdtd.set("dimension", '2d')
            fdtd.set("x min",-5e-6)
            fdtd.set("x max", 150e-6)
            fdtd.set("y min", -5e-6)
            fdtd.set("y max", 100e-6)
            fdtd.set("z", 0.0)
            fdtd.set("background material", "etch")
            
            fdtd.addrect()
            fdtd.set("name","TOX")
            fdtd.set("material","SiO2 (Glass) - Palik")
            fdtd.set("x min",-10e-6)
            fdtd.set("x max", 150e-6)
            fdtd.set("y min", 0.0)
            fdtd.set("y max", 3e-6)
            fdtd.set("z min", -1e-6)
            fdtd.set("z max", 1e-6)
            
            fdtd.addrect()
            fdtd.set("name","BOX")
            fdtd.set("material","SiO2 (Glass) - Palik")
            fdtd.set("x min",-10e-6)
            fdtd.set("x max", 150e-6)
            fdtd.set("y min", -3e-6)
            fdtd.set("y max", 0.0)
            fdtd.set("z min", -1e-6)
            fdtd.set("z max", 1e-6)
            
            fdtd.gdsimport(path+g+".GDS", g, 1, "Si3N4 (Silicon Nitride) - Phillip", -1.0e-6, 1.0e-6) #
            fdtd.set("name", g+"_base")
            fdtd.set("x", 0.0)
            fdtd.set("y", 0.0)
            fdtd.set("z", 0.0)
            
            
            if visualize_index == True:
                fdtd.addindex()
                fdtd.set("name","index_monitor")
                fdtd.set("monitor type",3)  # 2D y-normal
                fdtd.set("x min", 0.0)
                fdtd.set("x max", total_device_length)
                fdtd.set("y min", -5e-6)
                fdtd.set("y max", 5e-6)
                fdtd.set("z", 0.0)
                
                index = fdtd.getresult("index_monitor","index")
                ix = index['index_x']
                ix = ix[::10,::10,0,0]
                
                iy = index['index_y']
                iy = iy[::10,::10,0,0]
                index_abs = np.sqrt(np.abs(iy)**2)
                
                x1 = index["x"]
                x1 = x1[::10,0]
                x1 = [i*1000000 for i in x1]
                y1 = index["y"]
                y1 = y1[::10,0]
                y1 = [i*1000000 for i in y1]
                
                index_abs = index_abs.transpose()
                
                fig,ax=plt.subplots(1,1)
                cp=ax.contourf(x1,y1,index_abs, 200, zdir='z', offset=-100, cmap='jet')
                clb=fig.colorbar(cp)
                clb.ax.set_title('Index', fontweight="bold")
                for l in clb.ax.yaxis.get_ticklabels():
                    l.set_weight("bold")
                    l.set_fontsize(12)
                ax.set_xlabel('x-position (µm)', fontsize=13, fontweight="bold", labelpad=1)
                ax.set_ylabel('z-position (µm)', fontsize=13, fontweight="bold", labelpad=1)
                ax.xaxis.label.set_fontsize(13)
                ax.xaxis.label.set_weight("bold")
                ax.yaxis.label.set_fontsize(13)
                ax.yaxis.label.set_weight("bold")
                ax.tick_params(axis='both', which='major', labelsize=13)
                ax.set_yticklabels(ax.get_yticks(), weight='bold')
                ax.set_xticklabels(ax.get_xticks(), weight='bold')
                ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
                plt.title(g+'\n')
                plt.show()
                plt.close()
            
            fdtd.addport()
            fdtd.set("name","port 1")
            fdtd.set('x', 1.0e-6)
            fdtd.set('y min', -0.2e-6)
            fdtd.set('y max', 0.2e-6)
            fdtd.set('injection axis', 'x-axis')
            fdtd.set('direction', 'Forward')
            
            fdtd.setglobalsource("wavelength start",1092e-9)
            fdtd.setglobalsource("wavelength stop",1092e-9)
            fdtd.setglobalmonitor("frequency points",31)
            
            
            if full_power_distribution == True:
                fdtd.addpower()
                fdtd.set("name","E")
                fdtd.set("monitor type",7)
                fdtd.set("x min",-5e-6)
                fdtd.set("x max", 150e-6)
                fdtd.set("y min", -5e-6)
                fdtd.set("y max", 100e-6)
                fdtd.set("z", 0.0)
                
            fdtd.addpower()
            fdtd.set("name","E20")
            fdtd.set("monitor type",2)
            fdtd.set("x min",-5e-6)
            fdtd.set("x max", 150e-6)
            fdtd.set("y", 23e-6)
            fdtd.set("z", 0.0)
            
            fdtd.addpower()
            fdtd.set("name","E40")
            fdtd.set("monitor type",2)
            fdtd.set("x min",-5e-6)
            fdtd.set("x max", 150e-6)
            fdtd.set("y", 43e-6)
            fdtd.set("z", 0.0)
            
            fdtd.addpower()
            fdtd.set("name","E60")
            fdtd.set("monitor type",2)
            fdtd.set("x min",-5e-6)
            fdtd.set("x max", 150e-6)
            fdtd.set("y", 63e-6)
            fdtd.set("z", 0.0)
            
            fdtd.addpower()
            fdtd.set("name","E80")
            fdtd.set("monitor type",2)
            fdtd.set("x min",-5e-6)
            fdtd.set("x max", 150e-6)
            fdtd.set("y", 83e-6)
            fdtd.set("z", 0.0)
            
            fdtd.save('C:\\Users\\Lim Yudian\\Documents\\mixed_pitch_gratings\\script_run.fsp')
            print("simulating......")
            fdtd.run()
            print("simulation done!")
            
            if full_power_distribution == True:
                E = fdtd.getresult("E","E")
                E2 = E["E"]
                Ex1 = E2[:,:,:,0,0]
                Ey1 = E2[:,:,:,0,1]
                Ez1 = E2[:,:,:,0,2]
                Emag1 = np.sqrt(np.abs(Ex1)**2 + np.abs(Ey1)**2 + np.abs(Ez1)**2)
                Emag1 = Emag1[:,:,0]
                x1 = E["x"]
                x1 = x1[:,0]
                x1 = [i*1000000 for i in x1]
                y1 = E["y"]
                y1 = y1[:,0]
                y1 = [j*1000000 for j in y1]
                
                Emag1 = Emag1.transpose()
                Emag1_df = pd.DataFrame(Emag1)
                
                fig,ax=plt.subplots(1,1)
                cp=ax.contourf(x1,y1,Emag1, 200, zdir='z', offset=-100, cmap='jet')
                clb=fig.colorbar(cp)
                clb.ax.set_title('Electric Field (eV)', fontweight="bold")
                for l in clb.ax.yaxis.get_ticklabels():
                    l.set_weight("bold")
                    l.set_fontsize(12)
                ax.set_xlabel('x-position (µm)', fontsize=13, fontweight="bold", labelpad=1)
                ax.set_ylabel('y-position (µm)', fontsize=13, fontweight="bold", labelpad=1)
                ax.xaxis.label.set_fontsize(13)
                ax.xaxis.label.set_weight("bold")
                ax.yaxis.label.set_fontsize(13)
                ax.yaxis.label.set_weight("bold")
                ax.tick_params(axis='both', which='major', labelsize=13)
                ax.set_yticklabels(ax.get_yticks(), weight='bold')
                ax.set_xticklabels(ax.get_xticks(), weight='bold')
                ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
                plt.title(g+'\n')
                plt.show()
                plt.close()
                
            def getE(E_name, E_list):
                E = fdtd.getresult(E_name,"E")
                E2 = E["E"]
                Ex1 = E2[:,0,0,0,0]
                Ey1 = E2[:,0,0,0,1]
                Ez1 = E2[:,0,0,0,2]
                Emag1 = np.sqrt(np.abs(Ex1)**2 + np.abs(Ey1)**2 + np.abs(Ez1)**2)
                E_list.append(Emag1)
                x1 = E["x"]
                x1 = x1[:,0]
                x1 = [i*1000000 for i in x1]
                #plt.plot(x1, Emag1)
                #plt.title(E_name)
                #plt.show()
                #plt.close()
            
            getE('E20', E20_list)
            getE('E40', E40_list)
            getE('E60', E60_list)
            getE('E80', E80_list)
            small_pitch_list.append(small_pitch)
            big_pitch_list.append(big_pitch)
            small_count_list.append(small_count)
            big_count_list.append(big_count)
            small_dc_list.append(small_dc)
            big_dc_list.append(big_dc)
            
            df_ran_gds = pd.DataFrame()
            df_ran_gds['ran_gds'] = ran_gds_files
            df_ran_gds.to_csv(path+'ranlist.csv')
            ran_gds_files.append(g)
            
            
            fdtd.switchtolayout()
            fdtd.deleteall()
            #fdtd.save('C:\\Users\\Lim Yudian\\Documents\\mixed_pitch_gratings\\script_run.fsp')
          

            
            from IPython import get_ipython
            get_ipython().run_line_magic('matplotlib', 'inline')  # Reset plots
    
    df_results = pd.DataFrame()
    df_results['small_pitch_list'] = small_pitch_list
    df_results['big_pitch_list'] = big_pitch_list
    df_results['small_count_list'] = small_count_list
    df_results['big_count_list'] = big_count_list
    df_results['small_dc_list'] = small_dc_list
    df_results['big_dc_list'] = big_dc_list
    df_results['E20_list'] = E20_list
    df_results['E40_list'] = E40_list
    df_results['E60_list'] = E60_list
    df_results['E80_list'] = E80_list
    
    df_results.to_csv(path+'full_results'+str(n)+'.csv')