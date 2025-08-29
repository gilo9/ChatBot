import http.client
import json
from os import stat
import re
from urllib import response
import datetime
from arrow import get
from httpx import request

class APIFootball:

    def __init__(self):
        self.conn =http.client.HTTPSConnection("api.football-data.org") 
        self.headers = {
            'X-Auth-Token': "de2b1009a5f64220af15b15a86e6d0e0"
        }
        self.TeamIds={}

    def getStandings(self):

        self.conn.request("GET","/v4/competitions/PL/standings", headers=self.headers)

        data = self.getResponse()

        if data:
            responseString = "Premier League Standings\n"
            for team in data['standings'][0]['table']:
                responseString += f"{team['position']}. {team['team']['name']} {team['points']} points\n"
            return responseString
        else:
            return f"No table data found for the Premier League 24/25"

    def getMatches(self,team, status=None):
        id = self.getTeamId(team)
        if status == "last":
            self.conn.request("GET", f"/v4/teams/{id}/matches?status=FINISHED&limit=5", headers = self.headers)
        else:
            self.conn.request("GET", f"/v4/teams/{id}/matches?status=SCHEDULED&limit=5", headers = self.headers)

        data = self.getResponse()
        if data:
            responseString = ""
            for match in data["matches"]:
                responseString += f" {' at '.join(match['utcDate'].split('T'))} - {match['competition']['name']} : {match['homeTeam']['name']} vs {match['awayTeam']['name']} - {match['score']['fullTime']['home']}:{match['score']['fullTime']['away']} \n"
            return responseString
        else:
            return f"No match data found for {team}" 

    def getTeamIds(self):
        if len(self.TeamIds) <= 0:
            self.conn.request("GET", f"/v4/competitions/PL/teams?season=2024", headers = self.headers)
            
            data = self.getResponse()
            if data:
                for team in data['teams']:
                    self.TeamIds.update({team['name'].lower():team['id']})
            else:
                print("Error loading teams...")

    def getCoach(self,team):
        id = self.getTeamId(team)
        self.conn.request("GET", f"/v4/teams/{id}", headers = self.headers)
        data = self.getResponse()
        if data:
            return f"{team} is managed by {data['coach']['name']}"
        else:
            return f"No coach data found for {team}"
    
    def getTeam(self,team):
        teamId = self.getTeamId(team)
        if teamId == None:
            return "This team is not in the Premier Laeague"
        self.conn.request("GET", f"/v4/teams/{teamId}", headers = self.headers)
        responseString = ""
        data = self.getResponse()
        if data:
            responseString += "Team: " + data["name"] + "\n"
            responseString += "Coach: " + data["coach"]["name"] + "\n"
            for player in data["squad"]:
                responseString += f"{player['position']}: {player['name']} \n"
              #  kb.append(f"player({player['last_name']}) & plays_for({data['name']})")
            return responseString    
        else:
            return f"No Squad data found for {team}"

    def getTeamId(self,team):
        team = team.lower()
        try:
            if "fc" in team:
                teamId = self.TeamIds[team]
            else:
                teamId = self.TeamIds[f"{team} fc"]
            
        except KeyError as e:
           print(f"{team} fc is not in the premier league, try another team")
           teamId = None
        return teamId


    def getTeams(self):
        self.conn.request("GET", "/v4/teams/PL", headers = self.headers)
        return self.getResponse()

    def getScorers(self, limit):
        request = f"/v4/competitions/PL/scorers?limit={limit}"
        self.conn.request("GET", request, headers = self.headers)
        responseString = "Top Scorers\n"
        data = self.getResponse()
        if data:
            for scorer in data['scorers']:
                responseString += f"{scorer['player']['name']} ({scorer['team']['name']}) Goals: {scorer['goals']}, Assists : {scorer['assists'] }\n"
            return responseString
        else:
            return "I could not get the data, try again later"
        
    def getResponse(self):
        try:
            res = self.conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            if res.status == 200:
                return data
            else:
                return 
        except http.client.RemoteDisconnected as e:
            print("Error connecting to API, please try again later")





#api = APIFootball()

#api.getTeamIds()

#print(api.getTeam("Manchester United"))