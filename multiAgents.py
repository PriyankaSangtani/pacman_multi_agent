# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def euclideanDistance(self, xy1, xy2):
        "The Euclidean distance heuristic for a PositionSearchProblem"
        return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5  

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        #Extracting parameters of Successor Game State
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #Extracting parameters of Current Game State
        pos = successorGameState.getPacmanPosition()
        food = successorGameState.getFood()
        ghostStates = successorGameState.getGhostStates()
        scaredTime = [ghostState.scaredTimer for ghostState in ghostStates]
        
        '''Storing the location of food and initializing minimum distance of food and ghost to be infinity (999999) and 
        distance of scared ghost to be -infinity (-999999)
		'''
        foodlist = food.asList()
        minFoodDistance = 999999
        minGhostDistance = 999999
        maxScaredGhostDistance = -999999

        '''We choose the following weights for the different parameters
        Ghost = 2
        Ghost too close = 20
        Scared Ghost = 2
        Capsule = 20
        Food = 1.5
        '''
        ghostWeight = 2
        emergencyGhostWeight = 20
        scaredGhostWeight = 2
        capsuleLeftWeight = 20
        foodWeight = 1.5
        foodLeftWeight = 4

        #Getting the current score
        score = successorGameState.getScore()
		
		#Extracting minimum distance to food as per the state
        for food in foodlist:
            foodDistance = self.euclideanDistance(pos, food)

            if foodDistance > minFoodDistance:
                minFoodDistance = foodDistance

        minFoodDistance = max(0.3, minFoodDistance)

        #Extracting minimum distance to ghost and scared ghost as per the state
        ghostsScared = False
        for g in ghostStates:
            for i in range(len(ghostStates)):
                ghostDistance = self.euclideanDistance(pos, ghostStates[i].getPosition())
                if ghostDistance < minGhostDistance and g.scaredTimer <= 0:
                    minGhostDistance = ghostDistance
                    ghostsScared = False
                if ghostDistance > maxScaredGhostDistance and g.scaredTimer > 0:
                    maxScaredGhostDistance = ghostDistance
                    ghostsScared = True
    	minGhostDistance = max(0.3, minGhostDistance)
        
        if not ghostsScared:
            if minGhostDistance <= 3:
                score -= emergencyGhostWeight * float(1/minGhostDistance) #Decrease the score as pacman needs to go away if the ghost is too close

        if ghostsScared:
            maxScaredGhostDistance = max(0.3, maxScaredGhostDistance)
        
        score += scaredGhostWeight * float(1/maxScaredGhostDistance) #Increase the score as pacman needs to go close to scared ghost and try to eat it

        capsulesLeft = max(len(currentGameState.getCapsules()), 0.3)
        foodLeft = max(len(foodlist), 0.3)

        score += foodWeight * float(1/minFoodDistance) #Increase the score to make pacman go towards food

        #Increase the score to make pacman go towards capsule and left food
        score = score + capsuleLeftWeight * float(1/capsulesLeft)  + foodLeftWeight * float(1/foodLeft) 
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def isTerminal(self, state, depth, agent):
        if depth == self.depth or state.isWin() or state.isLose() or state.getLegalActions(agent) == 0:
            return True
        else:
            return False

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    #Definition of the minimax search
    def minimax(self, gameState, depth, agent):
        	
        	#Base case of the recursive function to check if terminal state is reached
            if self.isTerminal(gameState, depth, agent):
                return self.evaluationFunction(gameState)

            minMaxSuccessor = []

            #Run minimax for Pacman (agent = 0)
            if agent == 0:
                for newState in gameState.getLegalActions(agent):
                    minMaxSuccessor.append(self.minimax(gameState.generateSuccessor(agent, newState), depth, 1))
                return max(minMaxSuccessor)

            #Run minimax for the other agents (Ghosts)
            else:
                newAgent = agent + 1
                if gameState.getNumAgents() == newAgent:
                    newAgent = 0
                if newAgent == 0:
                    depth = depth + 1 #going to next level based on the agent
                for newState in gameState.getLegalActions(agent):
                    minMaxSuccessor.append(self.minimax(gameState.generateSuccessor(agent, newState), depth, newAgent))
                return min(minMaxSuccessor)

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        
        maximum = -999999
        for agentState in gameState.getLegalActions(0):
            utility = self.minimax(gameState.generateSuccessor(0, agentState), 0, 1)
            if utility > maximum or maximum == -999999: #Pruning the tree
                maximum = utility
                action = agentState

        return action

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    #Function to compute alpha value
    def alphaValue(self, gameState, depth, agent):
        
    	#Initializing Alpha to be -infinity
        alpha = -999999

        #Base case for the recursive function - check for terminal state
        if self.isTerminal(gameState, depth, agent):
            return self.evaluationFunction(gameState)

        #Exploring the state space
        for newState in gameState.getLegalActions(agent):
            alpha = max(alpha, self.betaValue(newState, depth + 1, agent))

        return alpha

    #Function to compute beta value
    def betaValue(self, gameState, depth, agent):

    	#Initializing Beta to be infinity
        beta = 999999

        #Base case for the recursive function - check for terminal state
        if self.isTerminal(gameState, depth, agent):
            return self.evaluationFunction(gameState)

        #Exploring the state space
        for newState in gameState.getLegalActions(agent):
            beta = min(beta, self.alphaValue(newState, depth + 1, agent))

        return beta

    #Function to explore state space using Alpha-Beta Pruning
    def minimax(self, gameState, depth, agent, alpha, beta):
        	
        #Base case for the recursive function - check for terminal state
        if self.isTerminal(gameState, depth, agent):
            return self.evaluationFunction(gameState)

        minMaxSuccessor = []

        #Run minimax for Pacman (agent = 0)
        if agent == 0:
            currentAlpha = -999999
            for newState in gameState.getLegalActions(agent):
                currentAlpha = max(currentAlpha, self.minimax(gameState.generateSuccessor(agent, newState), depth, 1, alpha, beta))
                if currentAlpha > beta:
                    return currentAlpha

                alpha = max(alpha, currentAlpha)
            return currentAlpha

        #Run minimax for the other agents (Ghosts)
        else:
            currentBeta = 999999
            newAgent = agent + 1
            if gameState.getNumAgents() == newAgent:
                newAgent = 0
            if newAgent == 0:
            	depth = depth + 1
            for newState in gameState.getLegalActions(agent):
                currentBeta = min(currentBeta, self.minimax(gameState.generateSuccessor(agent, newState), depth, newAgent, alpha, beta))
                if currentBeta < alpha:
                    return currentBeta
                beta = min(beta, currentBeta)

            return currentBeta


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        #Pruning of the tree based on Alpha, Beta  
        alpha = -999999
        beta = 999999
        utility = -999999
        for agentState in gameState.getLegalActions(0):
            utility = max(utility, self.minimax(gameState.generateSuccessor(0, agentState), 0, 1, alpha, beta))
            if utility > alpha:
                alpha = utility
                action = agentState

            alpha = max(alpha, utility)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    #Function to explore state space using Expectimax
    def minimax(self, gameState, depth, agent):
        
        #Base case for the recursive function - check for terminal state
        if self.isTerminal(gameState, depth, agent):
            return self.evaluationFunction(gameState)

        minMaxSuccessor = []
        #Run minimax for Pacman (agent = 0)
        if agent == 0:
            for newState in gameState.getLegalActions(agent):
               	minMaxSuccessor.append(self.minimax(gameState.generateSuccessor(agent, newState), depth, 1))
            return max(minMaxSuccessor)

        #Run minimax for the other agents (Ghosts)
        else:
            newAgent = agent + 1
            if gameState.getNumAgents() == newAgent:
               	newAgent = 0
            if newAgent == 0:
                depth = depth + 1
            for newState in gameState.getLegalActions(agent):
                minMaxSuccessor.append(self.minimax(gameState.generateSuccessor(agent, newState), depth, newAgent))

            # Here we take the expected value of the successor states by taking the average of the values of the sum of these states
            if len(minMaxSuccessor) != 0:
                return sum(minMaxSuccessor)/len(minMaxSuccessor)
            else:
                return 0

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        
                
        maximum = -999999
        for agentState in gameState.getLegalActions(0):
            utility = self.minimax(gameState.generateSuccessor(0, agentState), 0, 1)
            if utility > maximum or maximum == -999999:
                maximum = utility
                action = agentState

        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction