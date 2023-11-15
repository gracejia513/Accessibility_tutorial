class Accessibility_Seattle():
    def __init__(self,space_conv = 0.003, vot = 0.72, 
                 t_threshold = 15, 
                 FCA_pop_min = 0.05, 
                 delivery = 5,
                 beta = 86.9):
        self.beta = beta
        self.space_conv = space_conv
        self.sqm2sqft = 10.76
        self.delivery_distance = delivery
        self.delivery_start_distance = 1.5
        self.vot = vot
        self.radii = {}
        self.dist_dict = {}
        self.cost_dict = {}
        self.time_dict = {}
        self.poi_dict_AWOD = {}
        self.pop_dict_AWOD = {}
        self.poi_dict_AWDO = {}
        self.pop_dict_AWDO = {}
        self.poi_cap = {}
        self.pop2poi_AWOD = {}
        self.pop2poi_AWDO = {}
        self.R_j_AWOD = {}
        self.R_j_AWDO = {}
        self.pop_acc_AWOD = {}
        self.pop_acc_AWDO = {}
        self.pop_acc_AWD = {}
        self.food_cost = {}
        self.acc_AWOD = {}
        self.acc_AWD = {}
        self.acc_AWDO = {}
        self.div_index_AWOD = {}
        self.div_index_AWD = {}
        self.div_index_AWDO = {}
        self.walk_speed = 5
        self.drive_speed =40 #km/h
        self.t = t_threshold
        self.FCA_pop_min = FCA_pop_min
        self.D_j_AWOD = {}
        self.D_j_AWDO = {}
    
    def readPOI(self, path):
        import pandas as pd
        self.POI = pd.read_csv(path)
        for j in range(0,len(self.POI)):
            self.R_j_AWOD[j] = None
            self.R_j_AWDO[j] = None
            self.food_cost[j] = None
            self.D_j_AWOD[j] = None
            self.D_j_AWDO[j] = None
            self.pop_dict_AWOD[j] = []
            self.pop_dict_AWDO[j] = []
                   
        poi_types = set(self.POI["main_type"])
        self.poi_types = poi_types
        
        self.type_index = {} 
        for c in self.poi_types:
            self.type_index[c] = []
            if c == "restaurant":
                self.radii[c] = 6.1 
            elif c == "grocery":
                self.radii[c] = 4.67 
            else:
                self.radii[c] = 4.96 
            
            for j in range(0,len(self.POI)):
                if self.POI["main_type"][j] == c:
                    self.type_index[c].append(j)
        
    def readPOP(self,path, UD = False):
        import pandas as pd
        self.pop = pd.read_csv(path)
        self.tract_names  = {}
        self.tract_names_IP = {}
        
        for i in range(0, len(self.pop)):
            self.dist_dict[i] = {}  # 
            self.cost_dict[i] = {}  # dollar
            self.time_dict[i] = {}  # minutes
            self.pop2poi_AWOD[i] = {}
            self.pop2poi_AWDO[i] = {}
            self.tract_names_IP[i] = {}
            self.poi_dict_AWOD[i] = []
            self.poi_dict_AWDO[i] = []
            
            
        for i in range(0, len(self.pop)):
            
            if UD:
                if self.pop["NAMELSAD10"][i] == "Census Tract 53.01":
                    self.tract_names_IP[i] = "LIHP " + self.pop["NAMELSAD10"][i]
                elif self.pop["NAMELSAD10"][i] == "Census Tract 43.02":
                    self.tract_names_IP[i] = "LILP " + self.pop["NAMELSAD10"][i]
                elif self.pop["NAMELSAD10"][i] == "Census Tract 43.01":
                    self.tract_names_IP[i] = "HILP " + self.pop["NAMELSAD10"][i]
                else:
                    self.tract_names_IP[i] = "MIHP " + self.pop["NAMELSAD10"][i]  
            else:
                # self.tract_names_IP[i] = self.pop["LABEL"][i]
                self.tract_names_IP[i] = self.pop["NAMELSAD10"][i]
                
            self.pop_acc_AWOD[self.tract_names_IP[i]] = {}
            self.pop_acc_AWDO[self.tract_names_IP[i]] = {}
            self.pop_acc_AWD[self.tract_names_IP[i]] = {}
            for c in self.poi_types:
                self.pop_acc_AWOD[self.tract_names_IP[i]][c] =0
                self.pop_acc_AWDO[self.tract_names_IP[i]][c] =0
                self.pop_acc_AWD[self.tract_names_IP[i]][c] =0
                    
            self.acc_AWOD[self.tract_names_IP[i]] = None
            self.acc_AWD[self.tract_names_IP[i]] = None
        self.pop["MedianFamilyIncome"][84] = 18500
        
    def getCoords(self, loc_string):
        loc_dict = ast.literal_eval(loc_string)
        lat_NE = loc_dict['viewport']['northeast']['lat']
        lon_NE = loc_dict['viewport']['northeast']['lng']
        lat_SW = loc_dict['viewport']['southwest']['lat']
        lon_SW = loc_dict['viewport']['southwest']['lng']
        return [lat_NE,lon_NE,lat_SW,lon_SW]
    
    def getCapacity(self, 
                    lat_NE = False,lon_NE = False,lat_SW = False,lon_SW = False,
                    poly_wkt = False):
        import shapely
        import shapely.wkt
        from shapely.geometry.polygon import Polygon
        # return the square feet
        if poly_wkt:
            try:
                geom = shapely.wkt.loads(poly_wkt)
                geod = Geod(ellps="WGS84")
                area_sg = abs(geod.geometry_area_perimeter(geom)[0])
            except:
                # print("Could not create geometry because of errors while reading input.")
                area_sg = False
            

        if lat_NE:
            # build a polygon
            geom = Polygon([(lon_NE, lat_NE), 
                            (lon_SW, lat_NE), 
                            (lon_SW, lat_SW), 
                            (lon_NE, lat_SW),
                            (lon_NE, lat_NE)])

            geod = Geod(ellps="WGS84")
            area_gp = abs(geod.geometry_area_perimeter(geom)[0])
        # print('# Geodesic area: {:.3f} m^2'.format(area))
        if area_sg:
            return area_sg
        else:
            # this area is in sqft
            return area_gp * self.space_conv * self.sqm2sqft
        

    def addCapacity(self, GP = True):
        # service capacity as # indiv.
        if GP:
            for j in range(0,len(self.POI)):
                
                points = self.getCoords(self.POI["gp_geometr"][j])
                area = self.getCapacity(lat_NE = points[0],
                                           lon_NE = points[1],
                                           lat_SW = points[2],
                                           lon_SW = points[3],
                                           poly_wkt = self.POI["sg_g__poly"][j]) 
                
                food_type = self.POI['main_type'][j]
                
                if food_type == 'restaurant':
                    self.poi_cap[j] = area / 15
                    
                elif food_type == 'grocery':
                    self.poi_cap[j] = area / 20
                else:
                    self.poi_cap[j] = area / 12
        else:
            for j in range(0, len(self.POI)):
                self.poi_cap[j] = self.POI["capacity"][j] 
                
    def getManhattanDist(self, loc1_lat, loc1_lon, loc2_lat, loc2_lon):
        from haversine import haversine
        dist1 = haversine((loc1_lat,loc1_lon), (loc2_lat,loc1_lon))
        dist2 = haversine((loc2_lat,loc1_lon), (loc2_lat,loc2_lon))
        # haversine returns km, so this method converts and returns meters
        return (dist1 + dist2) * 1000
    
    def getMinkowskiDist(self,loc1_lat, loc1_lon, loc2_lat, loc2_lon):
        from haversine import haversine
        dist1 = haversine((loc1_lat,loc1_lon), (loc2_lat,loc1_lon))
        dist2 = haversine((loc2_lat,loc1_lon), (loc2_lat,loc2_lon))
        # haversine returns km, so this method converts and returns meters
        return ((dist1 * 1000) ** 1.54 + (dist2 * 1000) **1.54) ** (1/1.54)
    
    def getDistance(self, loc1_lat,loc1_lon,loc2_lat,loc2_lon):
        from haversine import haversine
        dist = haversine((loc1_lat,loc1_lon), (loc2_lat,loc2_lon))
        # haversine returns km, so this method converts and returns miles
        return dist/1.609

    
    def getDict(self, UD = False):
        import math
        for i in range(0, len(self.pop)):            
            for j in range(0,len(self.POI)):
                # food cost 
                delivery_cost = 0
                food_cost = 0
                
                if UD:
                    if self.POI["main_type"][j] == "grocery":
                        food_cost = 18.75
                    elif self.POI["main_type"][j] == "restaurant":
                        food_cost = 34.25
                    elif self.POI["main_type"][j] == "quick service":
                        food_cost = 20
                    else:
                        food_cost = 15
                    
                    if not math.isnan(self.POI["price_leve"][j]):
                        if self.POI["price_leve"][j] == 1.0:
                            food_cost_yelp = 10
        
                        elif self.POI["price_leve"][j] == 2.0:
                            food_cost_yelp = 22
                        elif self.POI["price_leve"][j] == 3.0:
                            food_cost_yelp = 35
                        food_cost = (food_cost + food_cost_yelp)/2
                        
                else:
                    food_cost = self.POI["price"][j]
        
                self.food_cost[j] = food_cost
                # transit_time_cost        
                # get minkowski distance
                # loc_1 pop
                # loc_2 POI
                
                if UD:
                    loc1_lat = self.pop["ycoord"][i]
                    loc1_lon = self.pop["xcoord"][i]
                    loc2_lat = self.POI["Latitude"][j]
                    loc2_lon = self.POI["Longitude"][j]
                    dist = self.getManhattanDist(loc1_lat, loc1_lon, loc2_lat, loc2_lon)
                else:
                    loc1_lat = self.pop["ycoord"][i]
                    loc1_lon = self.pop["xcoord"][i]
                    loc2_lat = self.POI["lat"][j]
                    loc2_lon = self.POI["lon"][j]
                    dist = self.getDistance(loc1_lat, loc1_lon, loc2_lat, loc2_lon)
                # this dist is in meters
                self.dist_dict[i][j] = dist
                # below is only needed if cost is included.
#                 if dist < 1609:
#                     #walking mode
#                     travel_time_hr = dist / 1000 / self.walk_speed
#                 else:
#                     # driving mode
#                     travel_time_hr = dist / 1000 / self.drive_speed
                
#                 self.time_dict[i][j] = travel_time_hr * 60
                
                
#                 travel_cost = travel_time_hr * self.pop["MedianFamilyIncome"][i]/8760 * self.vot
                
#                 #delivery cost
#                 #market share vs markup for fastfood vs markup for restaurant:
#                 # Doordash 0.59 0.46 0.47
#                 # Ubereats 0.24 0.91 0.49
#                 # Grubhub 0.14  0.25  0.37
#                 # Postmate 0.03  0.63 0.45
#                 # average markup = 0.54 vs 0.46
#                 if delivery_cost != 0:
#                     delivery_cost = food_cost * 0.5
                
#                 self.cost_dict[i][j] = {}
#                 self.cost_dict[i][j]["COST_AWOD"] = food_cost + travel_cost 
#                 self.cost_dict[i][j]["COST_AWDO"] = food_cost + delivery_cost
                
    def helper_getR(self):
        for i in range(0,len(self.pop)):
            # poi_list = []
            print("processing pop tract: {}".format(i))
            for j in range(0,len(self.POI)):
                # need to check the POI type
                poi_type = self.POI["main_type"][j]
                if self.dist_dict[i][j] <= self.radii[poi_type]:
                    # poi_list.append(j)
                    # without delivery
                    self.poi_dict_AWOD[i].append(j)
                    self.pop_dict_AWOD[j].append(i)
                if self.dist_dict[i][j] <= self.delivery_distance and self.dist_dict[i][j] >= self.delivery_start_distance:
                    # with delivery
                    self.poi_dict_AWDO[i].append(j)
                    self.pop_dict_AWDO[j].append(i)

                    
    def getDemand(self, case = 'On_premise'):
        import numpy as np
        # calculate how many residents will come to this poi j
        # For AWOD
        if case == "On_premise":
            for j in range(0,len(self.POI)):
                j_type = self.POI["main_type"][j]
                FCA_pop = 0
                for i in self.pop_dict_AWOD[j]:
                    total_poi = 0
                    for k in self.poi_dict_AWOD[i]:
                        if self.POI["main_type"][k] == j_type:
                            total_poi += np.exp(-self.dist_dict[i][k])

                    this_poi = np.exp(-self.dist_dict[i][j])
                    fraction = this_poi /total_poi
                    self.pop2poi_AWOD[i][j] = fraction
                    FCA_pop += self.pop["Pop2010"][i]* fraction

                FCA_pop = max(FCA_pop, self.FCA_pop_min)
                self.D_j_AWOD[j] = FCA_pop
        # AWDO        
        elif case == "Delivery":
            for j in range(0,len(self.POI)):
                j_type = self.POI["main_type"][j]
                FCA_pop = 0
                for i in self.pop_dict_AWDO[j]:
                    total_poi = 0
                    for k in self.poi_dict_AWDO[i]:
                        if self.POI["main_type"][k] == j_type:
                            total_poi += np.exp(-self.dist_dict[i][k])
                        else:
                            pass
                    this_poi = np.exp(-self.dist_dict[i][j])
                    fraction = this_poi / total_poi
                    self.pop2poi_AWDO[i][j] = fraction
                    FCA_pop += self.pop["Pop2010"][i]* fraction
                    
                 # get capacity
                FCA_pop = max(FCA_pop, self.FCA_pop_min)
                # this is actually AWDO
                self.D_j_AWDO[j] = FCA_pop
                
                
    def getR(self, case = 'On_premise'):
        import numpy as np
        # r is the supply/demand ratio, the supply capacity over total residents who wish to purchase food from poi j
        # AWOD
        if case == "On_premise":
            for j in range(0,len(self.POI)):
                j_type = self.POI["main_type"][j]
                FCA_pop = 0
                for i in self.pop_dict_AWOD[j]:
                    total_poi = 0
                    for k in self.poi_dict_AWOD[i]:
                        if self.POI["main_type"][k] == j_type:
                            total_poi += np.exp(-(self.dist_dict[i][k])**2 /self.beta)
                            # print(-self.cost_dict[i][k]["COST_AWOD"])
                            # print(total_poi)
                    this_poi = np.exp(-(self.dist_dict[i][j])**2 /self.beta)
                    if total_poi != 0:
                        fraction = this_poi / total_poi
                    else:
                        fraction = 0.0 

                    self.pop2poi_AWOD[i][j] = fraction
                    FCA_pop += self.pop["Pop2010"][i] * fraction

                FCA_pop = max(FCA_pop, self.FCA_pop_min)
                self.R_j_AWOD[j] = self.poi_cap[j]/FCA_pop
        # this is actually AWDO        
        elif case == "Delivery":
            for j in range(0,len(self.POI)):
                j_type = self.POI["main_type"][j]
                FCA_pop = 0
                for i in self.pop_dict_AWDO[j]:
                    total_poi = 0
                    # chek if this is AWOD or AWDO
                    for k in self.poi_dict_AWDO[i]:
                        if self.POI["main_type"][k] == j_type:
                            total_poi += np.exp(-(self.dist_dict[i][k])**2 /self.beta)
                        else:
                            pass
                    this_poi = np.exp(-(self.dist_dict[i][j])**2 /self.beta)
                    
                    fraction = this_poi /total_poi
                    self.pop2poi_AWDO[i][j] = fraction
                    FCA_pop += self.pop["Pop2010"][i] * fraction
                    
                 # get capacity
                FCA_pop = max(FCA_pop, self.FCA_pop_min)
                self.R_j_AWDO[j] = self.poi_cap[j]/FCA_pop
                
    def getR_v1(self, case = 'On_premise'):
        import numpy as np
        # r is the supply/demand ratio, the supply capacity over total residents who wish to purchase food from poi j
        # AWOD
        if case == "On_premise":
            for j in range(0,len(self.POI)):
                deno = 0
                for i in self.pop_dict_AWOD[j]:
                    deno += self.pop["Pop2010"][i] * np.exp(-(self.dist_dict[i][j])**2 /self.beta)

                self.R_j_AWOD[j] = self.poi_cap[j]/deno
        # this is actually AWDO        
        elif case == "Delivery":
            for j in range(0,len(self.POI)):
                deno = 0
                for i in self.pop_dict_AWDO[j]:
                    deno += self.pop["Pop2010"][i] * np.exp(-(self.dist_dict[i][j])**2 /self.beta)


                self.R_j_AWDO[j] = self.poi_cap[j]/deno            
                        
    def getAcc(self, case = "On_premise"):
        import numpy as np
        # poi_type = list(set(self.POI["main_type"]))
        if case == "On_premise":
            for i in range(0,len(self.pop)):
                for c in self.poi_types:
                    A_ic = 0
                    total_poi = 0
                    for j in self.poi_dict_AWOD[i]:
                        if self.POI["main_type"][j] == c:   
                            total_poi += np.exp(-self.dist_dict[i][j])                    
                    for j in self.poi_dict_AWOD[i]: 
                        if self.POI["main_type"][j] == c:
                            A_ic += self.R_j_AWOD[j] * (np.exp(-self.dist_dict[i][j]) / total_poi)
                    self.pop_acc_AWOD[self.tract_names_IP[i]][c] = A_ic
                    
        elif case == "Delivery":
            for i in range(0,len(self.pop)):
                for c in self.poi_types:
                    A_ic = 0
                    total_poi = 0
                    for j in self.poi_dict_AWDO[i]:                      
                        if self.POI["main_type"][j] == c:
                            total_poi += np.exp(-self.dist_dict[i][j])
                    for j in self.poi_dict_AWDO[i]:
                        if self.POI["main_type"][j] == c:
                            A_ic += self.R_j_AWDO[j] * np.exp(-self.dist_dict[i][j]) / total_poi
                    self.pop_acc_AWD[self.tract_names_IP[i]][c] =  A_ic + self.pop_acc_AWOD[self.tract_names_IP[i]][c]
                    self.pop_acc_AWDO[self.tract_names_IP[i]][c] = A_ic
                    
    def getDiversity(self, case = "On_premise"):
        import numpy as np
        A_i = []
        if case == "On_premise":
            for i in range(0,len(self.pop)):
                A_i.append(sum(self.pop_acc_AWOD[self.tract_names_IP[i]][c] for c in self.poi_types))
                d = 0
                for c in self.poi_types:
                    if A_i[i] !=0:
                        Q_ic = self.pop_acc_AWOD[self.tract_names_IP[i]][c] / A_i[i]
                    else:
                        Q_ic = 0
                    if Q_ic != 0:
                        d = d - Q_ic * np.log(Q_ic)
                self.acc_AWOD[self.tract_names_IP[i]] = A_i[i] ** d
                self.div_index_AWOD[self.tract_names_IP[i]] = d
        elif case == "Total":
            for i in range(0,len(self.pop)):
                A_i.append(sum(self.pop_acc_AWD[self.tract_names_IP[i]][c] for c in self.poi_types))
                d = 0
                for c in self.poi_types:
                    if A_i[i] !=0:
                        Q_ic = self.pop_acc_AWD[self.tract_names_IP[i]][c] / A_i[i]
                    else:
                        Q_ic = 0
                    if Q_ic != 0:
                        d = d - Q_ic * np.log(Q_ic)
                self.acc_AWD[self.tract_names_IP[i]] = A_i[i] ** d  
                self.div_index_AWD[self.tract_names_IP[i]] = d
        elif case == "Delivery":
            for i in range(0,len(self.pop)):
                A_i.append(sum(self.pop_acc_AWDO[self.tract_names_IP[i]][c] for c in self.poi_types))
                d = 0
                for c in self.poi_types:
                    if A_i[i] !=0:
                        Q_ic = self.pop_acc_AWDO[self.tract_names_IP[i]][c] / A_i[i]
                    else:
                        Q_ic = 0
                    if Q_ic != 0:
                        d = d - Q_ic * np.log(Q_ic)
                self.acc_AWDO[self.tract_names_IP[i]] = A_i[i] ** d
                self.div_index_AWDO[self.tract_names_IP[i]] = d
    def summary(self):
        import pandas as pd
        processed_tracts = {}
        # properties to be summarized include:
        # from acc.getAcc("cost + AWOD")
        # from acc.getAcc("cost + AWOD")
        # from acc.getDiversity("AWOD")
        # from acc.getDiversity("AWD")
        for i in range(0,len(self.pop)):
            processed_tracts[self.pop['NAMELSAD10'][i]] = {}
            for c in self.poi_types:
                processed_tracts[self.pop['NAMELSAD10'][i]]["Food Accessibility without Delivery " + c] = self.pop_acc_AWOD[self.tract_names_IP[i]][c]
                processed_tracts[self.pop['NAMELSAD10'][i]]["Food Accessibility with only Delivery " + c] = self.pop_acc_AWDO[self.tract_names_IP[i]][c]
                
            processed_tracts[self.pop['NAMELSAD10'][i]]["Diversity index without Delivery" ] = self.div_index_AWOD[self.tract_names_IP[i]]
            processed_tracts[self.pop['NAMELSAD10'][i]]["Diversity index with only Delivery" ] = self.div_index_AWDO[self.tract_names_IP[i]]
            processed_tracts[self.pop['NAMELSAD10'][i]]["Diversity index" ] = self.div_index_AWD[self.tract_names_IP[i]]  
            processed_tracts[self.pop['NAMELSAD10'][i]]["Diversity FA without Delivery" ] = self.acc_AWOD[self.tract_names_IP[i]]
            processed_tracts[self.pop['NAMELSAD10'][i]]["Diversity FA with only Delivery" ] = self.acc_AWDO[self.tract_names_IP[i]]
            processed_tracts[self.pop['NAMELSAD10'][i]]["Diversity FA" ] = self.acc_AWD[self.tract_names_IP[i]]
        df = pd.DataFrame(processed_tracts) 
        self.tracts_FA = df 
        df.to_csv("processed_tracts_with_Acc.csv")
        processed_pois = {}
        # properties to be summarized include:
        # from acc.getR("cost + AWOD")
        # from acc.getR("cost + AWOD")
        # from acc.getDemand("AWOD")
        # from acc.getDemand("AWD")
        for j in range(0,len(self.POI)):
            processed_pois[j] = {}
            processed_pois[j]["R without delivery"] = self.R_j_AWOD[j]
            processed_pois[j]["R with only delivery"] = self.R_j_AWDO[j]
            processed_pois[j]["Demand without delivery"] = self.D_j_AWOD[j]
            processed_pois[j]["Demand with only delivery"] = self.D_j_AWDO[j]
        df_poi = pd.DataFrame(processed_pois)            
        self.POIs_FA = pd.DataFrame(processed_pois)
        df_poi.to_csv("processed_POIs_with_R.csv")
        
        # for distance analysis
        # df_distance = {}
        for i in range(0,len(self.pop)):
            processed_tracts[self.pop['NAMELSAD10'][i]]["Available POI without Delivery "] = self.poi_dict_AWOD[i]
            processed_tracts[self.pop['NAMELSAD10'][i]]["Available POI with only Delivery "] = self.poi_dict_AWDO[i]
        df_distance = pd.DataFrame(processed_tracts)
        df_distance.to_csv("processed_tracts_with_Acc_and_distance.csv")