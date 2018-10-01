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

    def evaluationFunction2(self, currentGameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        pos = currentGameState.getPacmanPosition()
        food = currentGameState.getFood()
        ghostStates = currentGameState.getGhostStates()
        scaredTime = [ghostState.scaredTimer for ghostState in ghostStates]

        #newGhostPos = newGhostStates[0].getPosition()

        # print "Postion:", newPos
        # print "Ghost:", newGhostStates
        # print "Scared Time:", newScaredTimes
        # print "Ghost position:", newGhostPos

        foodlist = food.asList()
        minFoodDistance = 999999
        minGhostDistance = 999999
        minScaredGhostDistance = 0

        #print "Food List:", foodlist

        score = successorGameState.getScore()

        for foodnew in foodlist:
			foodDistance = util.manhattanDistance(pos, foodnew)
			if foodDistance < minFoodDistance:
				minFoodDistance = foodDistance

        for g in ghostStates:
        	for i in range(len(ghostStates)):
        		ghostDistance = util.manhattanDistance(pos, ghostStates[i].getPosition())
        		if ghostDistance < minGhostDistance:
        			if g.scaredTimer > 0:
    					minScaredGhostDistance = ghostDistance
        			else:
        				minGhostDistance = ghostDistance
        
        minGhostDistance = max(minGhostDistance, 5)
        score = score + (-1.5) * (1/minFoodDistance)			
        score = score + (-2) * (1/minGhostDistance) + (-2) * minScaredGhostDistance

        capsulesLeft = len(currentGameState.getCapsules())
        foodLeft = len(foodlist)

        score = score + (-20) * capsulesLeft + (-4) * foodLeft
        # print "Minimum Ghost Distance", minGhostDistance

        #score = (-1.5) * minFoodDistance + (-2) * (1/minGhostDistance) + (-2) * minScaredGhostDistance + (-20) * capsulesLeft + (-4) * foodLeft
        '''return successorGameState.getScore()'''
        return score

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        pos = successorGameState.getPacmanPosition()
        food = successorGameState.getFood()
        ghostStates = successorGameState.getGhostStates()
        scaredTime = [ghostState.scaredTimer for ghostState in ghostStates]

        #newGhostPos = newGhostStates[0].getPosition()

        # print "Postion:", newPos
        # print "Ghost:", newGhostStates
        # print "Scared Time:", newScaredTimes
        # print "Ghost position:", newGhostPos

        foodlist = food.asList()
        minFoodDistance = 999999
        minGhostDistance = 999999
        maxScaredGhostDistance = -999999

        ghostWeight = 2
        emergencyGhostWeight = 20
        scaredGhostWeight = 2
        capsuleLeftWeight = 20
        foodWeight = 1.5
        foodLeftWeight = 4

        #print "Food List:", foodlist

        score = successorGameState.getScore()
		
        for food in foodlist:
            foodDistance = self.euclideanDistance(pos, food)

            if foodDistance > minFoodDistance:
                minFoodDistance = foodDistance

        minFoodDistance = max(0.3, minFoodDistance)


        ghostsScared = False
        for g in ghostStates:
            # print g.scaredTimer
            for i in range(len(ghostStates)):
                ghostDistance = self.euclideanDistance(pos, ghostStates[i].getPosition())
                if ghostDistance < minGhostDistance and g.scaredTimer <= 0:
                    minGhostDistance = ghostDistance
                    ghostsScared = False
                if ghostDistance > maxScaredGhostDistance and g.scaredTimer > 0:
                    maxScaredGhostDistance = ghostDistance
                    ghostsScared = True
    	minGhostDistance = max(0.3, minGhostDistance)
        # score -= ghostWeight * 1/minGhostDistance

        if not ghostsScared:
            if minGhostDistance <= 3:
                score -= emergencyGhostWeight * float(1/minGhostDistance)
            # else:
            #     score -= ghostWeight * 1/minGhostDistance

        if ghostsScared:
            maxScaredGhostDistance = max(0.3, maxScaredGhostDistance)
            # scaredGhostWeight = foodLeftWeight * foodLeft/max(len(ghostStates), 0.1)
        
        score += scaredGhostWeight * float(1/maxScaredGhostDistance)

        capsulesLeft = max(len(currentGameState.getCapsules()), 0.3)
        foodLeft = max(len(foodlist), 0.3)

        score += foodWeight * float(1/minFoodDistance)

        #capsuleLeftWeight = foodLeftWeight * foodLeft/capsulesLeft
        score = score + capsuleLeftWeight * float(1/capsulesLeft)  + foodLeftWeight * float(1/foodLeft)
        # print "Minimum Ghost Distance", minGhostDistance

        #score = (-1.5) * minFoodDistance + (-2) * (1/minGhostDistance) + (-2) * minScaredGhostDistance + (-20) * capsulesLeft + (-4) * foodLeft
        '''return successorGameState.getScore()'''
        # print score
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def minimax(state, depth, agent):
    	if agent == state.getNumAgents:
    		return minimax(state, depth + 1, 0)

    	if self.isTerminal(state, depth, agent):
    		return self.evaluationFunction(state)
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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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

