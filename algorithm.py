from typing import List
import heapq


# Question 1 - DP Bottom Up
def fuse(fitmons: List[List[float]]) -> int:
    """
        Function description: 
            This function returns an integer, which is the maximum cuteness score that can be obtained by optimally fusing the fitmons.
            Fitmons are fused based their individual cuteness score and adjacency affinity.

        Approach description: 
            Since there are many possible ways to fuse the fitmons, we use 2D dynamic programming (DP) to systematically calculate and store the maximum score for all possible subsequences of fitmons to the table.
            memo[i][j] represents the maximum cuteness score achievable from fusing the subsequence of fitmons
            The table is initialized with the individual cuteness values of each fitmon as base cases.
            Then iteratively builds up solutions for longer subsequences by considering all possible fusions of shorter subsequences within them and finding the maximum value that we could obtain.
        
        Input: 
            fitmons: a list of lists, where inner list are represented as fitmons attributes [left_affinity, cuteness_score, right_affinity]

        Output:
            an integer represent the maximum cuteness score that can be achieved by considering all possible fusion of the fitmons.
            
        Time complexity:
            O(N + N^3) = O(N^3), 
            where N is the number of fitmons in the input list 'fitmons'

        Time complexity analysis: Given the N is the number of elements in the input list,

            Filling up the memo as the base case initialization with loop, takes O(N) time.

            Filling up the table section has 3 loops (nested):
            - The outermost loop runs for each possible length of subsequence (from 2 to n).
            - The second loop iterates through all possible starting points for these subsequences.
            - The innermost loop iterates through all possible partitions of the subsequence to consider different fusions.

            => a cubic time complexity.

        Space complexity: 
            O(N^2), where N is the number of fitmons
        
        Space complexity analysis:
            DP table uses 2D array which is a nested list to stores the maximum scores for every possible subsequence of 'fitmons' has a dimension of N*N = N^2,

    """

    length = len(fitmons)

    memo = [[-float('inf')] * length for _ in range(length)] 

    for i in range(len(fitmons)): #base case: initialize the first fitmons to its memo position ( O(N) )
        memo[i][i] = fitmons[i][1]
    
    # filling up the memo for longer subsequences or possibility fitmons fuse
    for n in range (2, length + 1): #length
        for i in range(length - n + 1):  #starting point of the current subsequence
            j = n + i - 1 #finding the ending point of current subsequence for partition.
            for partition in range (i,j): # every possible partition within this subsequence
                if memo[i][partition] != -float('inf') and memo[partition+1][j] != -float('inf'):
                    fuse_cuteness = int(memo[i][partition] * fitmons[partition][2] + memo[partition+1][j] * fitmons[partition+1][0])
                    memo[i][j] = max(memo[i][j], fuse_cuteness) #O(1)
                
    return memo[0][length-1]




# Q2 - dijkstra

class TreeMap:
    def __init__(self, roads: List[tuple[int,int,int]], solulus: List[tuple[int,int,int]]):
        """
            Function description: 
                Initialize TreeMap object with given roads and solulus details.

            Approach description:
                Initialize the graph by creating adjacency lists for each tree based on the road data.
                The maximum tree ID will be used to determine the adjacency list and Solulu list size.
                Each road is processed to populate the adjacency list, and each Solulu tuple is processed to update the Solulu list accordingly.

            Input:
                roads: A list of tuples where each tuple (u, v, w) represents a directed road from tree u to tree v with a required time w to travel.
                solulus: A list of tuples where each tuple (x, y, z) represents a Solulu tree at tree ID x, with y time to destroy the tree, and z as the tree ID to which one will be teleported if the Solulu tree is destroyed.

            Output:
                None, as it iss only the constructor of TreeMap class

            Time complexity: 
                O(|T| + |R|), where |T| is the number of unique trees and |R| is the number of roads.

            Time complexity analysis:
                Iterating through each road and Solulu entry costs O(|R| + |S|), where |S| is the number of Solulu entries.
                Since |S| is bounded by |T|, the complexity is linear with respect to the total number of trees and roads O(|T| + |R|).

            Space complexity: 
                O(|T| + |R|), where |T| is the number of unique trees and |R| is the number of roads.

            Space complexity analysis:
                The adjacency list storage requires O(|R|) aux space as it stores a list of connected vertex of each tree.
                The Solulu list requires O(|T|) aux space since each tree can have at most one associated Solulu action.
        
        """
        self.total_trees = 0
        for i in range (len(roads)): #finding the maximum number of tree ID
            x = max(roads[i][0], roads[i][1] )
            if x >= self.total_trees:
                self.total_trees = x +1

        
        self.graph = [[] for _ in range(self.total_trees + 1)] # initialize the array with empty list

        for i in range (len(roads)):
            u, v, w = roads[i]
            self.graph[u].append((v,w)) # append the connected tree to the table according to their index

        self.solulus = [(None, None) for i in range(self.total_trees + 1)] # initialize the array with tuples
        
        for i in range(len(solulus)): 
            x, y, z = solulus[i]
            self.solulus[x] = (y,z) # assign the value to the corresponding index


    def escape(self, start: int, exits: list[int]) -> tuple[int, list[int]]:
        """
            Function description: 
                Calculates the shortest time and path to escape from a starting tree to any designated exit tree.
                Utilizes a modified Dijkstra's algorithm to incorporate potential shortcuts via special Solulus trees.
           
            Approach description:
                The Dijkstra's algorithm uses a heapq to always extend the shortest discovered path. 
                It maintains a list of distances for each tree for both normal and Solulus-augmented conditions. 
                If a Solulus tree is encountered, it evaluates whether using it as a teleportation point offers a shortcut. 
                The function stops or return once the shortest path to any exit is found.
       
            Input:
                start: an integer represent the ID of the starting tree.
                exits: a list of integers (tree ID) represents the exit points.

            Output:
                This function returns an output of a tuple containing the minimum time required and its path to escape from a given starting tree to any designated exit tree, and after destroying 1 of the solulus tree.
                If no path to an exit is found, returns None.
            
            Time complexity:
                O(|T| + |R|log|T|) = O(|R|log|T|) time, where T is the set of unique trees in roads, and R is the set of roads.

            Time complexity analysis:
                The loop for the graph cost |R| since it iterates through all edges on each vertex or tree.
                Each vertex is pushed and popped from the heapq exactly once, and each edge is processed once. 
                The heapq operations are log|T|, leading to the complexity of  O(|T| + |R|log|T|) =  O(|R|log|T|).


            Space complexity: 
                O(|T| + |R|) auxiliary space, where |T| is the number of unique trees and |R| is the number of roads.

            Space complexity analysis:
                Requires space for the distance list which tracks distances under non-Solulu and Solulus-augmented conditions, plus the queue and path list.
        
        """
        if self.total_trees - 1 < start:
            return None
        
        shortest_path = self.dijkstra(start, exits)
        
        if shortest_path[1] == []: # if there are no paths from the start point to any exit points
            return None
        else:
            return shortest_path
    
    def dijkstra(self, start: int, exits: list[int]) -> tuple[int, list[int]]:
        """
            Function description: 
                Calculates the shortest path and time to reach the exit points by destroying 1 solulus tree.
                This function returns a tuple of integer and list (minimum_time, best_route).
                
            Input:
                start: an integer represent the ID of the starting tree.
                exits: a list of integers (tree ID) represents the exit points.

            Output:  
                This function returns an output of a tuple containing the minimum time required and its path to escape from a given starting tree to any designated exit tree by destroying 1 solulus tree.

            Time complexity:
                 O(|T| + |R|log|T|) = O(|R|log|T|) time, where T is the set of unique trees in roads, and R is the set of roads.

            Time complexity analysis:
                The loop for the graph cost |R| since it iterates through all edges on each vertex or tree.
                Each vertex is pushed and popped from the heapq exactly once, and each edge is processed once. 
                The heapq operations are log|T|, leading to the complexity of  O(|T| + |R|log|T|) =  O(|R|log|T|).

            Space complexity:
                O(|T| + |R|) auxiliary space, where |T| is the number of unique trees and |R| is the number of roads.

            Space complexity analysis:
                Requires space for the distance list which tracks distances under non-Solulu and Solulus-augmented conditions, also for the queue and path list.
        
        """

        distance_list = [] # O(|R|) aux space

        for _ in range (self.total_trees): 
            distance_list.append([float('inf'), float('inf')])
       
        distance_list[start][0] = 0
        minimum_time = float('inf')
        best_route = [] 

        q = [(0, start, False, [start])]

        while q:
            current_distance, u, has_clawed, paths = heapq.heappop(q) # O(log|T|)

            if has_clawed: # check if previously has clawed the solulus tree
                index = 1 
            else:
                index = 0

            # check if have clawed the solulus and the current vertex is in exit and is the minimum time.
            if has_clawed == True and u in exits and current_distance < minimum_time: # bounded by O(|T|) for exits
                minimum_time = current_distance # update the minimum time
                best_route = paths # update the best route
                continue

            if current_distance == distance_list[u][index]:
                for v, weight in self.graph[u]: # O(|R|)
                    path = paths + [v] # concatenate the path to the current list of paths
                    time = current_distance + weight

                    if time < distance_list[v][index]:
                        distance_list[v][index] = time # update the time if its lesser than the time in a current table
                        heapq.heappush(q, (time, v, has_clawed, path)) # O(log|T|)
                
                if self.solulus[u][0] != None and not has_clawed: # check whether the current vertex is solulu tree and not clawed yet
                    destroying_time, teleport = self.solulus[u]
                    total_time = current_distance + destroying_time

                    if total_time < distance_list[teleport][1]: # check if the time of clawing this tree is lesser than another tree that has been clawed and updated inside the table
                        if teleport != u:
                            teleport_path  = paths + [teleport] #concatenate the teleport path if it is not teleporting to the current tree.
                        else:
                            teleport_path = paths

                        distance_list[teleport][1] = total_time # update the table 
                        heapq.heappush(q, (total_time, teleport, True, teleport_path)) #O(log|T|)

        return (minimum_time , best_route)
