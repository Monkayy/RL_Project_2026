import gymnasium as gym


env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)

print("Osservazione iniziale:", observation)
print("Numero di azioni:", env.action_space.n)
print("Spazio delle osservazioni:", env.observation_space)

action = env.action_space.sample()

(
    next_observation,
    reward,
    terminated,
    truncated,
    info,
) = env.step(action)

print("Azione eseguita:", action)
print("Nuova osservazione:", next_observation)
print("Reward:", reward)
print("Terminato:", terminated)
print("Troncato:", truncated)

env.close()