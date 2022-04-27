import numpy as np
import zipfile
import json
import gym
import textworld.gym
import textworld
from tqdm import tqdm
import os
from os import listdir
from os.path import join
import gc

def generate_game(n_games=100):

    generated = 0
    while generated < n_games:
        seed = np.random.randint(2 ** 32 - 1)
        world_size = np.random.randint(5, 10)
        quest_length = np.random.randint(10, 30)
        n_object = np.random.randint(50, 100)
        print('generating....')
        os.system(f'tw-make custom --world-size {world_size} --quest-length {quest_length} --nb-objects {n_object} --output my_games/{seed}/game.ulx -f -v --seed {seed}')
        generated += 1


def info2graph(infos):
    graph = []
    for f in infos['facts']:
        f = f.serialize()
        sub = None
        rel = None
        obj = None
        name = f['name']
        args = f['arguments']
        if name == 'link':
            objs = [a['name'] for a in args]
            for i in range(len(objs) - 1):
                for j in range(1, len(objs)):
                    graph.append([objs[i], 'link', objs[j]])
        elif len(args) == 1:
            sub = args[0]['name']
            rel = 'is'
            obj = name
        elif len(args) == 2:
            sub = args[0]['name']
            rel = name
            obj = args[1]['name']
            graph.append([sub, rel, obj])
            if sub == 'P':
                sub = 'you'
        else:
            print('error: unknown type')
    return graph

def walk_thr(walks_per_game=10, max_walk=200):

    if not os.path.exists('game_walks'):
        os.mkdir('game_walks')

    walks = []
    games = [join('my_games', f, 'game.ulx') for f in listdir('my_games')]
    count = 0
    length = 0
    for g in (pbar := tqdm(games)):
        if length > 10000:
            game_walks_file = os.path.join('game_walks', 'game_walks_' + str(count) + '.json')
            with open(game_walks_file, 'w') as file:
                json.dump(walks, file)
                pbar.set_description("file saved: " + str(count + 1))
            count += 1
            walks = []
            length = 0
            gc.collect()
        for i in range(walks_per_game):
            walk = one_walk_thr(g, max_walk)
            length += len(walk)
            walks.append(walk)

    game_walks_file = os.path.join('game_walks', 'game_walks_' + str(count) + '.json')
    with open(game_walks_file, 'w') as file:
        json.dump(walks, file)
        print('complete!')
    
    

def one_walk_thr(game_dir, max_walk=200):
    request_infos = textworld.EnvInfos(admissible_commands=True, facts=True, policy_commands=True, verbs=True, entities=True)
    env_id = textworld.gym.register_game(game_dir, request_infos)
    env = gym.make(env_id)
    env.reset()
    initial = True
    walk = []
    obs = None
    done = False
    infos = None
    last_valid = False
    for i in range(max_walk):
        state = {}
        if (initial):
            obs, _, done, infos = env.step("look")
            prev_action = None
            initial = False
        else:
            last_state = walk[-1]
            if last_valid: # take a look after each policy action
                prev_action = 'look'
                last_valid = False
            if i % 5 == 0: # choose policy action
                last_valid = True
                policy = infos['policy_commands']
                prev_action = policy[0]
            elif i % 5 == 1: # choose an invalid action 
                verb = infos['verbs']
                entities = infos['entities']
                invalid_cmd = None
                while True:
                    vidx = np.random.randint(len(verb))
                    eidx = np.random.randint(len(entities))
                    invalid_cmd = verb[vidx] + ' ' + entities[eidx]
                    if invalid_cmd not in last_state['valid_actions']:
                        break
                prev_action = invalid_cmd
            else: # choose a random valid action
                idx = np.random.randint(len(last_state['valid_actions']))
                prev_action = last_state['valid_actions'][idx]
            obs, _, done, infos = env.step(prev_action)
        
        state['obs'] = obs
        state['valid_actions'] = infos["admissible_commands"]
        state['prev_action'] = prev_action
        state['complete_graph'] = info2graph(infos)
        walk.append(state)
        if done:
            break
    return walk

def unzip_games():
    if not os.path.exists('my_games'):
        os.mkdir('my_games')

    with zipfile.ZipFile('my_games.zip', 'r') as zip_ref:
        zip_ref.extractall('my_games')

def main():
    gc.enable()
    #unzip_games()
    #generate_game(1000)
    walk_thr()
    return 0

if __name__ == "__main__":
    main()