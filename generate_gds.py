# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:46:36 2025

@author: Lim Yudian
"""

import klayout.db as pya

small_pitchS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
big_pitchS = [0.5,0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
small_countS = [3,4,5,6,7,8,9,10]
big_countS = [3,4,5,6,7,8,9,10]
small_dcS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
big_dcS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
litho_limit = 0.1

for sp in small_pitchS:
    for bp in big_pitchS:
        for sc in small_countS:
            for bc in big_countS:
                for sdc in small_dcS:
                    for bdc in big_dcS:
                
                        small_pitch = sp
                        big_pitch = bp
                        small_count = sc
                        big_count = bc
                        small_dc = sdc
                        big_dc = bdc
                        thickness = 0.4
                        wg_length = 10
                        offset = 5
                        
                        if small_pitch>big_pitch or ((small_pitch*small_dc)<litho_limit) or ((1-(small_pitch*small_dc))<litho_limit) or ((big_pitch*big_dc)<litho_limit) or ((1-(big_pitch*big_dc))<litho_limit):
                            print("something wrong, no gds will be created")
                
                        
                        else:
                            print('gds will be created')
                            filename = 'sp'+str(int(small_pitch*1000))+'_'+'bp'+str(int(big_pitch*1000))+'_'+'sc'+str(int(small_count))+'_'+'bc'+str(int(big_count))+'_'+'sdc'+str(int(small_dc*1000))+'_'+'bdc'+str(int(big_dc*1000))+'_'+'end'
                            
                            # Create a new layout
                            layout = pya.Layout()
                            
                            # Create a new layer
                            layer_index = layout.layer(1, 0)  # Layer 1, datatype 0
                            
                            # Create a top-level cell
                            top_cell = layout.create_cell(filename)
                            
                            # waveguide
                            x1, y1 = 0, (-thickness/2)
                            x2, y2 = wg_length, (thickness/2)
                            rect = pya.DBox(x1, y1, x2, y2)  # Rectangle from (0, 0) to (1000, 2000)
                            # Convert to database units and add it to the layout
                            db_rect = pya.DPolygon(rect)  # Convert to database polygon
                            top_cell.shapes(layer_index).insert(db_rect)    
                            
                            #big pitch
                            for i in range(big_count):
                                x1, y1 = wg_length+((i+(1-big_dc))*big_pitch), (-thickness/2)
                                x2, y2 = x1+(big_pitch*big_dc), (thickness/2)
                                rect = pya.DBox(x1, y1, x2, y2)  # Rectangle from (0, 0) to (1000, 2000)
                                # Convert to database units and add it to the layout
                                db_rect = pya.DPolygon(rect)  # Convert to database polygon
                                top_cell.shapes(layer_index).insert(db_rect)  
                            
                            #small pitch
                            big_length = wg_length+(big_pitch*big_count)
                            for i in range(small_count):
                                x1, y1 = big_length+((i+(1-small_dc))*small_pitch), (-thickness/2)
                                x2, y2 = x1+(small_pitch*small_dc), (thickness/2)
                                rect = pya.DBox(x1, y1, x2, y2)  # Rectangle from (0, 0) to (1000, 2000)
                                # Convert to database units and add it to the layout
                                db_rect = pya.DPolygon(rect)  # Convert to database polygon
                                top_cell.shapes(layer_index).insert(db_rect)   
                            
                            #offsets
                            small_length = big_length+(small_pitch*small_count)
                            x1, y1 = small_length+((0+(1-small_dc))*small_pitch), (-thickness/2)
                            x2, y2 = x1+offset, (thickness/2)
                            rect = pya.DBox(x1, y1, x2, y2)  # Rectangle from (0, 0) to (1000, 2000)
                            # Convert to database units and add it to the layout
                            db_rect = pya.DPolygon(rect)  # Convert to database polygon
                            top_cell.shapes(layer_index).insert(db_rect) 
                            
                            
                            layout.write(filename+".gds")
                    
                    
                                                        