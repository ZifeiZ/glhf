#!/usr/bin/env python
# coding: utf-8

# # Twitter Data Crawling (Network Crawling)


get_ipython().system(u'pip install tweepy')
get_ipython().system(u'pip install networkx')

import tweepy

CONSUMER_KEY = "LTjScdEQ4VIKWGAzSAYhLXPYw"
CONSUMER_SECRET = "dyKwcKswtu2bO5E7EUrPW98S2A5wSuxCRhYgEcKrJdtJ0yHkdN"
ACCESS_TOKEN = "1221670631844012033-h3dx4L0GUGpXC25wgq2hwiaO1X2WdV"
ACCESS_TOKEN_SECRET = "OCYcuvgLc1euCfqUHTw2fGsaGHs2WiEZEtFlzdhfOzAUQ"


# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN,ACCESS_TOKEN_SECRET)

# build a api object, set the flag wait no rate limit to True
api = tweepy.API(auth,retry_count=5,retry_delay=1, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")


import pprint
rate_limit_status = api.rate_limit_status()
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(rate_limit_status)


limit_follower_ids = rate_limit_status['resources']['followers']['/followers/ids']
print("/followers/ids limit: {}, remaining: {}".format(limit_follower_ids["limit"],limit_follower_ids["remaining"]))


limit_friends_ids = rate_limit_status['resources']['friends']['/friends/ids']
print("/friends/ids limit: {}, remaining: {}".format(limit_friends_ids["limit"],limit_friends_ids["remaining"]))


# The function to crawl the users' friends 
def BFS(user_id, network, current_depth=0, max_depth=1):
    """
    :param user_id: the seed id we want to start with
    :param network: the dictionary we want to store the network
    :param current_depth: record the current depth 
    :param max_depth: the number of hops we want to creal start from the seed id
    :return: 
    """
    if(current_depth > max_depth):
        return 
    
    # crawl the friends of the user
    friends_id_list = set()
    for friends_id in tweepy.Cursor(api.friends_ids,user_id=user_id,count=5000).items():
        friends_id_list.add(friends_id)
    network[user_id] = friends_id_list
    print("In depth: {}, {} friends ids are crawled".format(current_depth,len(friends_id_list)))
    
    depth = current_depth+1
    for friends_id in friends_id_list:
        BFS(friends_id,network,depth,max_depth)


def tweepy_authentication(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET):
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    # build a api object, set the flag wait no rate limit to True
    api = tweepy.API(auth, retry_count=5, retry_delay=1, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    try:
        api.verify_credentials()
        print("Authentication OK")
    except:
        print("Error during authentication")

    return api


# authenticate multiple apis, which are stored in api_list
f = open('token_list.txt')
api_list = [] # the api objects are stored in api_list
for line in f:
    CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET = line.strip().split(' ')
    api_list.append(tweepy_authentication(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET))



import os
import json
import time

def BFS_queue(user_id, save_path, api_list, max_depth):
    """
    :param user_id: the seed user id we want to start with
    :param save_path: for each user, we will save the user's friendlist
    :param api_list: an array of api objects
    :param max_depth: the maximual depth we want to use to crawl the network
    :return: 
    """
    queue = []
    queue.append((user_id, 0))
    current_depth = 0

    if (not os.path.exists(save_path)):
        os.makedirs(save_path)

    limit_friends_number_list = []
    for api in api_list:
        rate_limit_status = api.rate_limit_status()
        limit_friends_number_list.append(rate_limit_status['resources']['friends']['/friends/ids']["remaining"])

    rate_limit_exceeded = False

    while (len(queue) > 0):
        head_user_id, current_depth = queue.pop()

        if (current_depth > max_depth):
            break

        friends_id_list = []
        
        # check whether reach the rate limit
        rate_limit_exceeded = True
        for i in range(0, len(api_list)):
            if limit_friends_number_list[i] > 0:
                rate_limit_exceeded = False
                break
        current_api_id = i
        if rate_limit_exceeded:
            print("Rate limit reached, we need to sleep 900 s.")
            time.sleep(910)
            # after sleep, reset to 15
            for k in range(0, len(api_list)):
                limit_friends_number_list[k] = 15
                rate_limit_exceeded = False
                current_api_id = 0

        # start crawling
        api = api_list[current_api_id]
        for friends_id in tweepy.Cursor(api.friends_ids, user_id=head_user_id, count=5000).pages():
            friends_id_list.extend(friends_id)
            limit_friends_number_list[current_api_id] -= 1

            # check whether reach the rate limit
            rate_limit_exceeded = True
            for i in range(0, len(api_list)):
                if limit_friends_number_list[i] > 0:
                    rate_limit_exceeded = False
                    break
            current_api_id = i
            if rate_limit_exceeded:
                print("Rate limit reached, we need to sleep 900 s.")
                time.sleep(910)
                # after sleep, reset to 15
                for k in range(0, len(api_list)):
                    limit_friends_number_list[k] = 15
                    rate_limit_exceeded = False
                    current_api_id = 0
                api = api_list[current_api_id]

        print("In depth: {}, {} friends ids are crawled".format(current_depth, len(friends_id_list)))

        friends_id_dict = {"user_id": head_user_id, "friends_id": friends_id_list}

        # save to file
        with open(os.path.join(save_path, "{}.json".format(head_user_id)), 'w') as f:
            json.dump(friends_id_dict, f)

        if (current_depth < max_depth):
            queue.extend([(friends_id, current_depth + 1) for friends_id in friends_id_list])


def post_process(network, convert_to_undirected=True):
    # confrim the nodes of the network
    user_ids = set(network.keys())
    
    # eliminate the stored friends ids that are not covered in the nodes users ids of the network
    for user,friends in network.items():
        friends = user_ids&friends
        network[user] = friends    
        
    # convert to undirected
    if convert_to_undirected:
        for user,friends in network.items():
            for friend in friends:
                network[friend].add(user)
        for user,friends in network.items():
            network[user] = list(friends)
    return network


# ## Post process for the second version of BFS


def post_process_2(save_path, convert_to_undirected=True):
    network = dict()
    file_names = os.listdir(save_path)
    user_ids = set()
    
    # confrim the nodes of the network
    for name in file_names:
        user_id = int(name.replace(".json",""))
        user_ids.add(user_id)
        
    # eliminate the stored friends ids that are not covered in the nodes users ids of the network
    for name in file_names:
        friends_id_dict = json.load(open(os.path.join(save_path,name),"r"))
        
        user_id = friends_id_dict["user_id"]
        friends_id_set = set(friends_id_dict["friends_id"]) & user_ids

        network[user_id] = friends_id_set
        
    # convert to undirected
    if convert_to_undirected:
        for user,friends in network.items():
            for friend in friends:
                network[friend].add(user)

        for user,friends in network.items():
            network[user] = list(friends)
    return network


# # Get the user id of Trump
screen_name = "realDonaldTrump"
user = api.get_user(screen_name)
user_id = user.id


# # build network with breadth-first-search
network = dict()

# # Since the network only care about Trump and Trump's friends, we set the max_depth as 1
BFS(user_id,current_depth=0,max_depth=1,network=network)

network = post_process(network)


# Get the user id of Trump
screen_name = "realDonaldTrump"
user = api.get_user(screen_name)
user_id = user.id

# build network with breadth-first-search
network = dict()

# set the save location for the jsons
save_path = "./results/network_medium"

# use queue to do BFS
BFS_queue(user_id=user_id, api_list=api_list, max_depth=1, save_path=save_path)

# post process for the second version
network = post_process_2(save_path=save_path)


directory = "./results"
if(not os.path.exists(directory)):
    os.mkdir(directory)
    print("{} has been added".format(directory))
else:
    print("{} already existed".format(directory))
    
network_path = "./results/network.json"
with open(network_path,"w") as f:
    json.dump(network,f)
    print("The network has been saved in {}.".format(network_path))


# Get the user id of mine
screen_name = "ZifeiZ"
user = api.get_user(screen_name)
user_id = user.id

# build network with breadth-first-search
network = dict()

# set the save location for the jsons
save_path = "./results/network_medium"

# use queue to do BFS
BFS_queue(user_id=user_id, api_list=api_list, max_depth=1, save_path=save_path)

# post process for the second version
network = post_process_2(save_path=save_path)

directory = "./results"
if(not os.path.exists(directory)):
    os.mkdir(directory)
    print("{} has been added".format(directory))
else:
    print("{} already existed".format(directory))
    
network_path = "./results/Zifei_Zheng_network.json"
with open(network_path,"w") as f:
    json.dump(network,f)
    print("The network has been saved in {}.".format(network_path))


# With this function we can get the names of the users in network
def get_names(network):
    user_ids = list(network.keys())
    profiles = api.lookup_users(user_ids=user_ids)
    friend_names = dict()
    for profile in profiles:
        friend_names[profile.id]=profile.name
    return friend_names

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import networkx as nx

def network_visualize(network, friend_names):
    figure = plt.figure(figsize=(20,14))

    G = nx.Graph()
    for user_id,friend_ids in network.items():
        for friend_id in friend_ids:
            G.add_edge(friend_names[user_id],friend_names[friend_id])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, font_size=10, with_labels=False)
    for p in pos:  # raise text positions
        pos[p][1] += 0.07
    nx.draw_networkx_labels(G, pos)
    plt.show()

## network visualization

# load the network
network_path = "./results/network.json"
with open(network_path) as f:
    my_network = json.load(f)
    
# convert string key to int key
mynetwork = dict()
for key in my_network:
    mynetwork[int(key)] = my_network[key]

# 1. get the user names 
friend_names = get_names(mynetwork)

# 2. visualize the network
network_visualize(mynetwork,friend_names)

# load the network
my_network_path = "./results/Zifei_Zheng_network.json"
with open(my_network_path) as f:
    _my_network = json.load(f)
    
# convert string key to int key
_mynetwork = dict()
for key in _my_network:
    _mynetwork[int(key)] = _my_network[key]

# 1. get the user names 
my_friend_names = get_names(_mynetwork)

# 2. visualize the network
network_visualize(_mynetwork,my_friend_names)