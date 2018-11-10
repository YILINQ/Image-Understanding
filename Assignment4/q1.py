import CNN
import robot
import gameState as game


def system():
    while not game.GameOver:
        while game.isPlaying:
            robot.wait()
        robot.scanCourt()
        objects = robot.detect(CNN)
        location = robot.compute3dLocation(objects["ball"])
        if game.ballInsideThisCourt:
            robot.moveToLocation(location)
            robot.grabBall(location)
            if game.thisPlayerServing:
                # if this player is going to serve, perfect.
                # send the ball to the player
                # ------ scan detect compute and move to location
                robot.scanCourt()
                objects = robot.detect(CNN)
                location = robot.compute3dLocation(objects["player"])
                robot.moveToLocation(location)
                # ------ scan detect compute and move to location
                robot.leaveCourt()
                robot.faceLocation(game.otherSide)
                robot.victoryDanceAtLocation(robot.currentLocation)
                robot.wait()
            else:
                # grabbed the ball but player not serving, send it to supply
                # ------ scan detect compute and move to location
                robot.scanCourt()
                objects = robot.detect(CNN)
                location = robot.compute3dLocation(objects["supply"])
                robot.moveToLocation(location)
                # ------ scan detect compute and move to location
                robot.leaveCourt()
                robot.faceLocation(game.otherSide)
                robot.victoryDanceAtLocation(robot.currentLocation)
                robot.wait()
        else:
            if game.thisPlayerServing:
                # the player is going to serve but no ball detected
                # go to the supply and grab a ball
                # send the ball to player and leave the court
                robot.scanCourt()
                objects = robot.detect(CNN)
                location = robot.compute3dLocation(objects["supply"])
                robot.moveToLocation(location)
                robot.grabBall(location)

                robot.scanCourt()
                objects = robot.detect(CNN)
                location = robot.compute3dLocation(objects["player"])
                robot.moveToLocation(location)
                robot.grabBall(location)

                robot.leaveCourt()
                robot.faceLocation(game.otherSide)
                robot.victoryDanceAtLocation(robot.currentLocation)
                robot.wait()
            else:
                # ball is not in this half court, just wait
                robot.wait()
