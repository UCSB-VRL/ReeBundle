#parameters
#eps: for connect disconnect
#tau: for interruption
import numpy as np
import pdb

class Event:
    def __init__(self, event, trajectory = None, t = None):
        self.event = event
        self.trajectory = trajectory
        self.t = t

def checkEpsilonDistance(p1, p2, eps):
	return (np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)) <= eps

def findConnectDisconnectEvents(t1_id, t2_id, t1, t2, eps):
	dic_t1 = {}
	dic_t2 = {}
	ti = 0
	flag_t1 = [False]*len(t1)
	flag_t2 = [False]*len(t2)
	while ti < (len(t1)):
		tj = 0 
		while tj < (len(t2)):
			if not flag_t1[ti] and not flag_t2[tj] and checkEpsilonDistance(t1[ti], t2[tj], eps):
				# print(ti,"Start",tj)
				flag_t1[ti] = True
				flag_t2[tj] = True
				flag_insert = True
				first_i = ti
				last_i = ti
				first_j = tj
				last_j = tj
				while(flag_insert and last_j<len(t2) and first_j>=0):
					# print("flag_t1",flag_t1)
					# print("flag_t2", flag_t2)
					# print("(" ,first_i, " " , last_i , ") (" , first_j ,  " " , last_j ,")")
					flag_insert = False
					#case first_i - 1 insert:
					if (first_i - 1 >=0) and not flag_t1[first_i - 1 ]:
						if first_j - 1 >= 0 and not flag_t2[first_j - 1] and checkEpsilonDistance(t1[first_i - 1], t2[first_j - 1], eps):
							first_i = first_i - 1
							first_j = first_j - 1
							flag_t1[first_i ] = True
							flag_t2[first_j ] = True
							flag_insert = True
						elif last_j + 1 < len(t2) and not flag_t2[last_j + 1] and checkEpsilonDistance(t1[first_i - 1], t2[last_j + 1], eps):
							first_i = first_i - 1
							last_j = last_j + 1
							flag_t1[first_i ] = True
							flag_t2[last_j ] = True
							flag_insert = True
						elif first_j + 1 < len(t2) and not flag_t2[first_j + 1] and checkEpsilonDistance(t1[first_i - 1], t2[first_j + 1], eps):
							first_i = first_i - 1
							first_j = first_j + 1
							flag_t1[first_i ] = True
							flag_t2[first_j ] = True
							flag_insert = True
						elif last_j - 1 >= 0 and not flag_t2[last_j - 1] and checkEpsilonDistance(t1[first_i - 1], t2[last_j - 1], eps):
							first_i = first_i - 1
							last_j = last_j - 1
							flag_t1[first_i ] = True
							flag_t2[last_j ] = True
							flag_insert = True
						else:
							for each_j in range(first_j, last_j + 1):
								if checkEpsilonDistance(t1[first_i - 1], t2[each_j], eps):
									first_i = first_i - 1
									flag_t1[first_i ] = True
									flag_t2[each_j] = True
									flag_insert = True
									break


									
					#case last_i + 1 insert:
					if (last_i + 1 < len(t1)) and not flag_t1[last_i + 1]:
						if first_j - 1 > 0 and  not flag_t2[first_j - 1] and checkEpsilonDistance(t1[last_i + 1], t2[first_j - 1], eps):
							last_i = last_i + 1
							first_j = first_j - 1
							flag_t1[last_i ] = True
							flag_t2[first_j ] = True 
							flag_insert = True
						elif last_j + 1 < len(t2) and not flag_t2[last_j + 1] and checkEpsilonDistance(t1[last_i + 1], t2[last_j + 1], eps):
							last_i = last_i + 1
							last_j = last_j + 1
							flag_t1[last_i ] = True
							flag_t2[last_j ] = True
							flag_insert = True
						elif first_j + 1 < len(t2) and not flag_t2[first_j + 1] and checkEpsilonDistance(t1[last_i + 1], t2[first_j + 1], eps):
							last_i = last_i + 1
							first_j = first_j + 1
							flag_t1[last_i ] = True
							flag_t2[first_j] = True
							flag_insert = True
						elif last_j - 1 > 0 and not flag_t2[last_j -1] and checkEpsilonDistance(t1[last_i + 1], t2[last_j - 1], eps):
							last_i = last_i + 1
							last_j = last_j - 1
							flag_t1[last_i ] = True
							flag_t2[last_j ] = True
							flag_insert = True
						else:
							for each_j in range(first_j, last_j + 1):
								if checkEpsilonDistance(t1[last_i + 1], t2[each_j], eps):
									last_i = last_i + 1
									flag_t1[last_i ] = True
									flag_t2[each_j] = True
									flag_insert = True
									break
						


									
					#case first_j - 1 insert:
					if (first_j - 1 >= 0 ) and not flag_t2[first_j - 1]:
						if first_i - 1 >= 0 and not flag_t1[first_i - 1 ] and checkEpsilonDistance(t2[first_j - 1], t1[first_i - 1], eps):
							first_j = first_j - 1
							first_i = first_i - 1
							flag_t1[first_i ] = True
							flag_t2[first_j] = True
							flag_insert = True
						elif last_i + 1 < len(t1) and not flag_t1[last_i + 1] and checkEpsilonDistance(t2[first_j - 1], t1[last_i + 1], eps):
							first_j = first_j - 1
							last_i = last_i + 1
							flag_t1[last_i ] = True
							flag_t2[first_j ] = True
							flag_insert = True
						elif first_i + 1 < len(t1) and not flag_t1[first_i + 1] and checkEpsilonDistance(t2[first_j - 1], t1[first_i + 1], eps):
							first_j = first_j - 1
							first_i = first_i + 1
							flag_t2[first_j ] = True
							flag_t1[first_i ] = True
							flag_insert = True
						elif last_i - 1 >= 0 and not flag_t1[last_i - 1] and checkEpsilonDistance(t2[first_j - 1], t1[last_i - 1], eps):
							first_j = first_j - 1
							last_i = last_i - 1
							flag_t1[last_i ] = True
							flag_t2[first_j] = True
							flag_insert = True

						else:
							for each_i in range(first_i, last_i + 1):
								if checkEpsilonDistance(t2[first_j - 1], t1[each_i], eps):
									first_j = first_j - 1
									flag_t2[first_j] = True
									flag_t1[each_i] = True
									flag_insert = True
									break
						
									

					#case last_j + 1 insert:
					if (last_j + 1 < len(t2)) and not flag_t2[last_j + 1]:
						if  first_i - 1 >= 0 and not flag_t1[first_i - 1] and checkEpsilonDistance(t2[last_j + 1], t1[first_i - 1], eps):
							last_j = last_j + 1
							first_i = first_i - 1
							flag_t1[first_i ] = True
							flag_t2[last_j ] = True
							flag_insert = True
						elif last_i + 1 < len(t1) and not flag_t1[last_i + 1] and checkEpsilonDistance(t2[last_j + 1], t1[last_i + 1], eps):
							last_j = last_j + 1
							last_i = last_i + 1
							flag_t1[last_i ] = True
							flag_t2[last_j ] = True
							flag_insert = True
						elif  first_i + 1 <len(t1) and not flag_t1[first_i + 1] and checkEpsilonDistance(t2[last_j + 1], t1[first_i + 1], eps):
							last_j = last_j + 1
							first_i = first_i + 1
							flag_t1[first_i ] = True
							flag_t2[last_j ] = True
							flag_insert = True
						elif last_i - 1 >= 0 and not flag_t1[last_i - 1] and checkEpsilonDistance(t2[last_j + 1], t1[last_i - 1], eps):
							last_j = last_j + 1
							last_i = last_i - 1
							flag_t1[last_i ] = True
							flag_t2[last_j ] = True
							flag_insert = True
						else:
							for each_i in range(first_i, last_i + 1):
								if checkEpsilonDistance(t2[last_j + 1], t1[each_i], eps):
									last_j = last_j + 1
									flag_t2[last_j] = True
									flag_t1[each_i] = True									
									flag_insert = True
									break
# 				print("Connected segment t1 and t2 at (" ,first_i, " " , last_i , ") (" , first_j ,  " " , last_j ,")", t1_id, t2_id)						
				# print("Connected segment t1 and t2 at (" first_i " " last_i ") (" first_j  " " last_j ")", ti, tj)
				#connect event
				if dic_t1.get(first_i):
					dic_t1 [first_i].append(Event("connect", t2_id, first_j))
				else:
					dic_t1 [first_i] = [Event("connect", t2_id, first_j)]

				if dic_t2.get(first_j):
					dic_t2 [first_j].append(Event("connect", t1_id, first_i))
				else:
					dic_t2 [first_j] = [Event("connect", t1_id, first_i)]
				#disconnect event
				if last_i + 1 < len(t1): #beacuse it gets disconnected later
					if dic_t1.get(last_i + 1):
						dic_t1 [last_i + 1].append(Event("disconnect", t2_id, last_j + 1))
					else:
						dic_t1 [last_i + 1] = [Event("disconnect", t2_id, last_j + 1)]
				if  last_j + 1 < len(t2):
					if dic_t2.get(last_j + 1):
						dic_t2 [last_j + 1].append(Event("disconnect", t1_id, last_i + 1))
					else:
						dic_t2 [last_j + 1] = [Event("disconnect", t1_id, last_i + 1)]

					




				ti = last_i				
				break
			else:
				tj += 1		
		ti += 1
# 	print(dic_t1, dic_t2)	
	return dic_t1, dic_t2	
	# if any of the above takes place repeat
	#nothing happened then terminate







