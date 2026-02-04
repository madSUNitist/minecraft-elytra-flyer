import numpy as np


# def reward_stage_1(self, old_distance, dpitch_raw, dyaw_raw, should_add_firework, terminated, truncated):
#     reward = 0.0
#     current_distance = np.linalg.norm(self.target_position - self.entity.position)
    
#     # 1. distance reduction reward
#     distance_reduction = old_distance - current_distance # -infty ~ +1.7
    
#     distance_factor = 0.1 + 0.9 * (current_distance / max(self.initial_distance, 1.0)) # 0.0 ~ 1.0
#     reward += distance_reduction * distance_factor * 0.5 # -infty ~ 0.85
    
#     # 2. velocity orientation reward
#     if np.linalg.norm(self.entity.velocity) > 0.1:
#         to_target = self.target_position - self.entity.position
#         to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
#         velocity_dir = self.entity.velocity / (np.linalg.norm(self.entity.velocity) + 1e-6)
#         direction_similarity = np.dot(velocity_dir, to_target_norm) # -1.0 ~ 1.0
#         reward += direction_similarity * 0.01 # -0.01 ~ 0.01
    
#     # 3. firework punishment
#     if should_add_firework:
#         # basic punishment
#         reward -= 0.05
#         # reward -= 0.1
#         self.total_fireworks_used += 1
#         # # additional punishment
#         # rocket_penalty_factor = 1.0 + 0.1 * self.total_fireworks_used
#         # reward -= 0.05 * rocket_penalty_factor
    
#     # 4. distance punishment
#     # if current_distance > self.initial_distance * 1.2:  # 20%
#     #     excess_factor = (current_distance - self.initial_distance * 1.2) / self.initial_distance
#     #     reward -= excess_factor * 0.5
    
#     # * d_pitch penalty
#     reward -= 0.004 * dpitch_raw ** 2 # -0.001 ~ 0.001
    
#     # * d_yaw penalty
#     reward -= 0.1 * dyaw_raw ** 2     # -0.001 ~ 0.001
    
#     # 5. hit
#     if terminated and current_distance < 5.0:  # hit
#         # fall damage
#         fall_damage = self.entity.get_fall_damage()
#         # clip
#         capped_damage = min(fall_damage, 20.0) # 0.0 ~ 20.0
        
#         # basic reward
#         base_success_reward = 100.0
        
#         # damage
#         damage_penalty = -capped_damage
        
#         # # firework efficiency
#         # rocket_efficiency_bonus = max(0, 30.0 - 5.0 * self.total_fireworks_used)
        
#         # time efficiency
#         time_efficiency_bonus = max(0, 20.0 - 0.2 * self.current_step)
        
#         # totall reward
#         success_reward = (base_success_reward + 
#                             damage_penalty + 
#                         #  rocket_efficiency_bonus + 
#                             time_efficiency_bonus)
        
#         # clip
#         success_reward = max(success_reward, 1.0)
        
#         reward += success_reward
        
#         # info
#         self._last_success_info = {
#             "distance": current_distance,
#             "fall_damage": fall_damage,
#             "capped_damage": capped_damage,
#             "rockets_used": self.total_fireworks_used,
#             "steps": self.current_step,
#             "time": self.total_time,
#             "base_reward": base_success_reward,
#             "damage_penalty": damage_penalty,
#             # "rocket_bonus": rocket_efficiency_bonus,
#             "time_bonus": time_efficiency_bonus,
#             "total_success_reward": success_reward
#         }
    
#     # 6. oob punishment
#     elif terminated:
#         reward -= 50.0
    
#     # 7. step punishment
#     reward -= 0.0002 * self.current_step
    
#     # 8. alive reward
#     reward += 0.0005
    
#     return reward

# def reward_stage_2_1(self, old_distance, dpitch_raw, dyaw_raw, should_add_firework, terminated, truncated):
#     reward = 0.0
#     current_distance = np.linalg.norm(self.target_position - self.entity.position)
    
#     # 1. distance reduction reward
#     distance_reduction = old_distance - current_distance # -infty ~ +1.7
    
#     distance_factor = 0.1 + 0.9 * (current_distance / max(self.initial_distance, 1.0)) # 0.0 ~ 1.0
#     reward += distance_reduction * distance_factor * 0.5 # -infty ~ 0.85
    
#     # 2. velocity orientation reward
#     if np.linalg.norm(self.entity.velocity) > 0.1:
#         to_target = self.target_position - self.entity.position
#         to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
#         velocity_dir = self.entity.velocity / (np.linalg.norm(self.entity.velocity) + 1e-6)
#         direction_similarity = np.dot(velocity_dir, to_target_norm) # -1.0 ~ 1.0
#         reward += direction_similarity * 0.01 # -0.01 ~ 0.01
    
#     # 3. firework punishment
#     if should_add_firework:
#         # basic punishment
#         reward -= 0.05
#         # reward -= 0.1
#         self.total_fireworks_used += 1
#         # additional punishment
#         rocket_penalty_factor = 1.0 + 0.1 * self.total_fireworks_used
#         reward -= 0.05 * rocket_penalty_factor
    
#     # 4. distance punishment
#     # if current_distance > self.initial_distance * 1.2:  # 20%
#     #     excess_factor = (current_distance - self.initial_distance * 1.2) / self.initial_distance
#     #     reward -= excess_factor * 0.5
    
#     # * d_pitch penalty
#     reward -= 0.004 * dpitch_raw ** 2 # -0.001 ~ 0.001
    
#     # * d_yaw penalty
#     reward -= 0.1 * dyaw_raw ** 2     # -0.001 ~ 0.001
    
#     # 5. hit
#     success_reward = 0.0
#     if terminated and current_distance < 5.0:  # hit
#         # fall damage
#         fall_damage = self.entity.get_fall_damage()
#         # clip
#         capped_damage = min(fall_damage, 20.0) # 0.0 ~ 20.0
        
#         # basic reward
#         base_success_reward = 100.0
        
#         # damage
#         damage_penalty = -capped_damage * 2.0 # 0.0 ~ 40.0
        
#         # firework efficiency
#         rocket_efficiency_bonus = max(0, 30.0 - 5.0 * self.total_fireworks_used)
        
#         # time efficiency
#         time_efficiency_bonus = max(0, 20.0 - 0.2 * self.current_step)
        
#         # totall reward
#         success_reward = (
#             base_success_reward + 
#             damage_penalty + 
#             rocket_efficiency_bonus + 
#             time_efficiency_bonus
#         )
        
#         # clip
#         success_reward = max(success_reward, 1.0)
        
#         # info
#         self._last_success_info = {
#             "distance": current_distance,
#             "fall_damage": fall_damage,
#             "capped_damage": capped_damage,
#             "rockets_used": self.total_fireworks_used,
#             "steps": self.current_step,
#             "time": self.total_time,
#             "base_reward": base_success_reward,
#             "damage_penalty": damage_penalty,
#             "rocket_bonus": rocket_efficiency_bonus,
#             "time_bonus": time_efficiency_bonus,
#             "total_success_reward": success_reward
#         }
    
#     # 6. oob punishment
#     elif terminated:
#         reward -= 10.0
    
#     # 7. step punishment
#     reward -= 0.0002 * self.current_step
    
#     # 8. alive reward
#     # reward += 0.0005
    
#     reward = np.clip(reward, -10, 10) + success_reward
    
#     return reward

# def reward_stage_2_2(self, old_distance, dpitch_raw, dyaw_raw, should_add_firework, terminated, truncated):
#     reward = 0.0
#     current_distance = np.linalg.norm(self.target_position - self.entity.position)
    
#     # 1. distance reduction reward
#     distance_reduction = old_distance - current_distance # -infty ~ +1.7
    
#     distance_factor = 0.1 + 0.9 * (current_distance / max(self.initial_distance, 1.0)) # 0.0 ~ 1.0
#     reward += distance_reduction * distance_factor * 0.5 # -infty ~ 0.85
    
#     # 2. velocity orientation reward
#     if np.linalg.norm(self.entity.velocity) > 0.1:
#         to_target = self.target_position - self.entity.position
#         to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
#         velocity_dir = self.entity.velocity / (np.linalg.norm(self.entity.velocity) + 1e-6)
#         direction_similarity = np.dot(velocity_dir, to_target_norm) # -1.0 ~ 1.0
#         reward += direction_similarity * 0.01 # -0.01 ~ 0.01
    
#     # 3. firework punishment
#     if should_add_firework:
#         # basic punishment
#         reward -= 0.05
#         # reward -= 0.1
#         self.total_fireworks_used += 1
#         # additional punishment
#         rocket_penalty_factor = 1.0 + 0.1 * self.total_fireworks_used
#         reward -= 0.05 * rocket_penalty_factor
    
#     # 4. distance punishment
#     # if current_distance > self.initial_distance * 1.2:  # 20%
#     #     excess_factor = (current_distance - self.initial_distance * 1.2) / self.initial_distance
#     #     reward -= excess_factor * 0.5
    
#     # * d_pitch penalty
#     reward -= 0.04 * dpitch_raw ** 2 # -0.001 ~ 0.001
    
#     # * d_yaw penalty
#     reward -= 1.0 * dyaw_raw ** 2     # -0.001 ~ 0.001
    
#     # 5. hit
#     success_reward = 0.0
#     if terminated and current_distance < 5.0:  # hit
#         # fall damage
#         fall_damage = self.entity.get_fall_damage()
#         # clip
#         capped_damage = min(fall_damage, 20.0) # 0.0 ~ 20.0
        
#         # basic reward
#         base_success_reward = 100.0
        
#         # damage
#         damage_penalty = -capped_damage * 2.0 # 0.0 ~ 40.0
        
#         # firework efficiency
#         rocket_efficiency_bonus = max(0, 30.0 - 5.0 * self.total_fireworks_used)
        
#         # time efficiency
#         time_efficiency_bonus = max(0, 20.0 - 0.2 * self.current_step)
        
#         # totall reward
#         success_reward = (
#             base_success_reward + 
#             damage_penalty + 
#             rocket_efficiency_bonus + 
#             time_efficiency_bonus
#         )
        
#         # clip
#         success_reward = max(success_reward, 1.0)
        
#         # info
#         self._last_success_info = {
#             "distance": current_distance,
#             "fall_damage": fall_damage,
#             "capped_damage": capped_damage,
#             "rockets_used": self.total_fireworks_used,
#             "steps": self.current_step,
#             "time": self.total_time,
#             "base_reward": base_success_reward,
#             "damage_penalty": damage_penalty,
#             "rocket_bonus": rocket_efficiency_bonus,
#             "time_bonus": time_efficiency_bonus,
#             "total_success_reward": success_reward
#         }
    
#     # 6. oob punishment
#     elif terminated:
#         reward -= 10.0
    
#     # 7. step punishment
#     reward -= 0.0002 * self.current_step
    
#     # 8. alive reward
#     # reward += 0.0005
    
#     reward = np.clip(reward, -10, 10) + success_reward
    
#     return reward


# def reward_stage_2_3(self, old_distance, dpitch_raw, dyaw_raw, should_add_firework, terminated, truncated):
#     reward = 0.0
#     current_distance = np.linalg.norm(self.target_position - self.entity.position)
    
#     # 1. distance reduction reward
#     distance_reduction = old_distance - current_distance # -infty ~ +1.7
    
#     distance_factor = 0.1 + 0.9 * (current_distance / max(self.initial_distance, 1.0)) # 0.0 ~ 1.0
#     reward += distance_reduction * distance_factor * 0.5 # -infty ~ 0.85
    
#     # 2. velocity orientation reward
#     if np.linalg.norm(self.entity.velocity) > 0.1:
#         to_target = self.target_position - self.entity.position
#         to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
#         velocity_dir = self.entity.velocity / (np.linalg.norm(self.entity.velocity) + 1e-6)
#         direction_similarity = np.dot(velocity_dir, to_target_norm) # -1.0 ~ 1.0
#         reward += direction_similarity * 0.01 # -0.01 ~ 0.01
    
#     # 3. firework punishment
#     if should_add_firework:
#         # basic punishment
#         reward -= 0.05
#         # reward -= 0.1
#         self.total_fireworks_used += 1
#         # additional punishment
#         rocket_penalty_factor = 1.0 + 0.1 * self.total_fireworks_used
#         reward -= 0.05 * rocket_penalty_factor
    
#     # 4. distance punishment
#     # if current_distance > self.initial_distance * 1.2:  # 20%
#     #     excess_factor = (current_distance - self.initial_distance * 1.2) / self.initial_distance
#     #     reward -= excess_factor * 0.5
    
#     # * d_pitch penalty
#     reward -= 0.4 * dpitch_raw ** 2 # -0.001 ~ 0.001
    
#     # * d_yaw penalty
#     reward -= 10.0 * dyaw_raw ** 2     # -0.001 ~ 0.001
    
#     # 5. hit
#     success_reward = 0.0
#     if terminated and current_distance < 5.0:  # hit
#         # fall damage
#         fall_damage = self.entity.get_fall_damage()
#         # clip
#         capped_damage = min(fall_damage, 20.0) # 0.0 ~ 20.0
        
#         # basic reward
#         base_success_reward = 100.0
        
#         # damage
#         damage_penalty = -capped_damage * 2.0 # 0.0 ~ 40.0
        
#         # firework efficiency
#         rocket_efficiency_bonus = max(0, 30.0 - 5.0 * self.total_fireworks_used)
        
#         # time efficiency
#         time_efficiency_bonus = max(0, 20.0 - 0.2 * self.current_step)
        
#         # totall reward
#         success_reward = (
#             base_success_reward + 
#             damage_penalty + 
#             rocket_efficiency_bonus + 
#             time_efficiency_bonus
#         )
        
#         # clip
#         success_reward = max(success_reward, 1.0)
        
#         # info
#         self._last_success_info = {
#             "distance": current_distance,
#             "fall_damage": fall_damage,
#             "capped_damage": capped_damage,
#             "rockets_used": self.total_fireworks_used,
#             "steps": self.current_step,
#             "time": self.total_time,
#             "base_reward": base_success_reward,
#             "damage_penalty": damage_penalty,
#             "rocket_bonus": rocket_efficiency_bonus,
#             "time_bonus": time_efficiency_bonus,
#             "total_success_reward": success_reward
#         }
    
#     # 6. oob punishment
#     elif terminated:
#         reward -= 10.0
    
#     # 7. step punishment
#     reward -= 0.0002 * self.current_step
    
#     # 8. alive reward
#     # reward += 0.0005
    
#     reward = np.clip(reward, -10, 10) + success_reward
    
#     return reward


# def reward_stage_3(self, old_distance, dpitch_raw, dyaw_raw, should_add_firework, terminated, truncated):
#     reward = 0.0
#     current_distance = np.linalg.norm(self.target_position - self.entity.position)
    
#     # 1. distance reduction reward
#     distance_reduction = old_distance - current_distance # -infty ~ +1.7
    
#     distance_factor = 0.1 + 0.9 * (current_distance / max(self.initial_distance, 1.0)) # 0.0 ~ 1.0
#     reward += distance_reduction * distance_factor * 0.5 # -infty ~ 0.85
    
#     # 2. velocity orientation reward
#     if np.linalg.norm(self.entity.velocity) > 0.1:
#         to_target = self.target_position - self.entity.position
#         to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
#         velocity_dir = self.entity.velocity / (np.linalg.norm(self.entity.velocity) + 1e-6)
#         direction_similarity = np.dot(velocity_dir, to_target_norm) # -1.0 ~ 1.0
#         reward += direction_similarity * 0.01 # -0.01 ~ 0.01
    
#     # 3. firework punishment
#     if should_add_firework:
#         # basic punishment
#         reward -= 0.05
#         # reward -= 0.1
#         self.total_fireworks_used += 1
#         # additional punishment
#         rocket_penalty_factor = 1.0 + 0.1 * self.total_fireworks_used
#         reward -= 0.05 * rocket_penalty_factor
    
#     # 4. distance punishment
#     # if current_distance > self.initial_distance * 1.2:  # 20%
#     #     excess_factor = (current_distance - self.initial_distance * 1.2) / self.initial_distance
#     #     reward -= excess_factor * 0.5
    
#     # * d_pitch penalty
#     reward -= 0.4 * dpitch_raw ** 2 # -0.001 ~ 0.001
    
#     # * d_yaw penalty
#     reward -= 10.0 * dyaw_raw ** 2     # -0.001 ~ 0.001
    
#     # 5. hit
#     success_reward = 0.0
#     if terminated and current_distance < 5.0:  # hit
#         # fall damage
#         fall_damage = self.entity.get_fall_damage()
#         # clip
#         capped_damage = min(fall_damage, 20.0) # 0.0 ~ 20.0
        
#         # basic reward
#         base_success_reward = 100.0
        
#         # damage
#         damage_penalty = -capped_damage * 2.0 # 0.0 ~ 40.0
        
#         # firework efficiency
#         rocket_efficiency_bonus = max(0, 30.0 - 5.0 * self.total_fireworks_used)
        
#         # time efficiency
#         time_efficiency_bonus = max(0, 20.0 - 0.2 * self.current_step)
        
#         # totall reward
#         success_reward = (
#             base_success_reward + 
#             damage_penalty + 
#             rocket_efficiency_bonus + 
#             time_efficiency_bonus
#         )
        
#         # clip
#         success_reward = max(success_reward, 1.0)
        
#         # info
#         self._last_success_info = {
#             "distance": current_distance,
#             "fall_damage": fall_damage,
#             "capped_damage": capped_damage,
#             "rockets_used": self.total_fireworks_used,
#             "steps": self.current_step,
#             "time": self.total_time,
#             "base_reward": base_success_reward,
#             "damage_penalty": damage_penalty,
#             "rocket_bonus": rocket_efficiency_bonus,
#             "time_bonus": time_efficiency_bonus,
#             "total_success_reward": success_reward
#         }
    
#     # 6. oob punishment
#     elif terminated:
#         reward -= 10.0
    
#     # 7. step punishment
#     reward -= 0.0002 * self.current_step
    
#     # 8. alive reward
#     # reward += 0.0005
    
#     reward = np.clip(reward, -10, 10) + success_reward
    
#     return reward


# def reward_stage_4(self, old_distance, dpitch_raw, dyaw_raw, should_add_firework, terminated, truncated):
#     reward = 0.0
#     current_distance = np.linalg.norm(self.target_position - self.entity.position)
    
#     # 1. distance reduction reward
#     # distance_reduction = old_distance - current_distance # -infty ~ +1.7
    
#     # distance_factor = 0.1 + 0.9 * (current_distance / max(self.initial_distance, 1.0)) # 0.0 ~ 1.0
#     # reward += distance_reduction * distance_factor * 0.5 # -infty ~ 0.85
    
#     # 2. velocity orientation reward
#     # if np.linalg.norm(self.entity.velocity) > 0.1:
#     #     to_target = self.target_position - self.entity.position
#     #     to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
#     #     velocity_dir = self.entity.velocity / (np.linalg.norm(self.entity.velocity) + 1e-6)
#     #     direction_similarity = np.dot(velocity_dir, to_target_norm) # -1.0 ~ 1.0
#     #     reward += direction_similarity * 0.01 # -0.01 ~ 0.01
    
#     # 3. firework punishment
#     if should_add_firework:
#         # basic punishment
#         reward -= 0.05
#         self.total_fireworks_used += 1
#         # additional punishment
#         rocket_penalty_factor = 1.0 + 0.1 * self.total_fireworks_used
#         reward -= 0.05 * rocket_penalty_factor
    
#     # 4. distance punishment
#     # if current_distance > self.initial_distance * 1.2:  # 20%
#     #     excess_factor = (current_distance - self.initial_distance * 1.2) / self.initial_distance
#     #     reward -= excess_factor * 0.5
    
#     # * d_pitch penalty
#     reward -= 0.2 * dpitch_raw ** 2 # -0.1 ~ 0.1
    
#     # * d_yaw penalty
#     reward -= 5.0 * dyaw_raw ** 2     # -0.1 ~ 0.1
    
#     # 5. hit
#     success_reward = 0.0
#     if terminated:  # hit
#         # fall damage
#         fall_damage = self.entity.get_fall_damage()
#         # clip
#         capped_damage = min(fall_damage, 20.0) # 0.0 ~ 20.0
        
#         # basic reward
#         base_success_reward = 500.0
        
#         # damage
#         damage_penalty = -capped_damage * 5.0 # 0.0 ~ 100.0
        
#         # firework efficiency
#         rocket_efficiency_bonus = max(0, 30.0 - 5.0 * self.total_fireworks_used)
        
#         # time efficiency
#         time_efficiency_bonus = max(0, 20.0 - 0.2 * self.current_step)
        
#         # totall reward
#         success_reward = (
#             base_success_reward + 
#             damage_penalty + 
#             rocket_efficiency_bonus + 
#             time_efficiency_bonus
#         )
        
#         # clip
#         success_reward = max(success_reward, 1.0)
        
#         # info
#         self._last_success_info = {
#             "distance": current_distance,
#             "fall_damage": fall_damage,
#             "capped_damage": capped_damage,
#             "rockets_used": self.total_fireworks_used,
#             "steps": self.current_step,
#             "time": self.total_time,
#             "base_reward": base_success_reward,
#             "damage_penalty": damage_penalty,
#             "rocket_bonus": rocket_efficiency_bonus,
#             "time_bonus": time_efficiency_bonus,
#             "total_success_reward": success_reward
#         }
    
#     # 6. oob punishment
#     elif terminated:
#         reward -= 10.0
    
#     # 7. step punishment
#     reward -= 0.01 * self.current_step
    
#     # 8. alive reward
#     # reward += 0.0005
    
#     reward = np.clip(reward, -10, 10) + success_reward
    
#     return reward * 0.1

# def reward_stage_5(self, old_distance, dpitch_raw, dyaw_raw, should_add_firework, terminated, truncated):
#     reward = 0.0
#     current_distance = np.linalg.norm(self.target_position - self.entity.position)
    
#     # 1. distance reduction reward
#     # distance_reduction = old_distance - current_distance # -infty ~ +1.7
    
#     # distance_factor = 0.1 + 0.9 * (current_distance / max(self.initial_distance, 1.0)) # 0.0 ~ 1.0
#     # reward += distance_reduction * distance_factor * 0.5 # -infty ~ 0.85
    
#     # 2. velocity orientation reward
#     # if np.linalg.norm(self.entity.velocity) > 0.1:
#     #     to_target = self.target_position - self.entity.position
#     #     to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
#     #     velocity_dir = self.entity.velocity / (np.linalg.norm(self.entity.velocity) + 1e-6)
#     #     direction_similarity = np.dot(velocity_dir, to_target_norm) # -1.0 ~ 1.0
#     #     reward += direction_similarity * 0.01 # -0.01 ~ 0.01
    
#     # 3. firework punishment
#     if should_add_firework:
#         # basic punishment
#         reward -= 0.5
#         self.total_fireworks_used += 1
#         # additional punishment
#         rocket_penalty_factor = 1.0 + 0.1 * self.total_fireworks_used
#         reward -= 0.5 * rocket_penalty_factor
    
#     # 4. distance punishment
#     # if current_distance > self.initial_distance * 1.2:  # 20%
#     #     excess_factor = (current_distance - self.initial_distance * 1.2) / self.initial_distance
#     #     reward -= excess_factor * 0.5
    
#     # * d_pitch penalty
#     reward -= 0.2 * dpitch_raw ** 2 # -0.1 ~ 0.1
    
#     # * d_yaw penalty
#     reward -= 5.0 * dyaw_raw ** 2     # -0.1 ~ 0.1
    
#     # 5. hit
#     success_reward = 0.0
#     if terminated:  # hit
#         # fall damage
#         fall_damage = self.entity.get_fall_damage()
#         # clip
#         capped_damage = min(fall_damage, 20.0) # 0.0 ~ 20.0
        
#         # basic reward
#         base_success_reward = 400.0
        
#         # damage
#         damage_penalty = -capped_damage * 5.0 # 0.0 ~ 100.0
        
#         # firework efficiency
#         rocket_efficiency_bonus = max(0, 30.0 - 5.0 * self.total_fireworks_used)
        
#         # time efficiency
#         time_efficiency_bonus = max(0, 20.0 - 0.2 * self.current_step)
        
#         # totall reward
#         success_reward = (
#             base_success_reward + 
#             damage_penalty + 
#             rocket_efficiency_bonus + 
#             time_efficiency_bonus
#         )
        
#         # clip
#         success_reward = max(success_reward, 1.0)
        
#         # info
#         self._last_success_info = {
#             "distance": current_distance,
#             "fall_damage": fall_damage,
#             "capped_damage": capped_damage,
#             "rockets_used": self.total_fireworks_used,
#             "steps": self.current_step,
#             "time": self.total_time,
#             "base_reward": base_success_reward,
#             "damage_penalty": damage_penalty,
#             "rocket_bonus": rocket_efficiency_bonus,
#             "time_bonus": time_efficiency_bonus,
#             "total_success_reward": success_reward
#         }
    
#     # 6. oob punishment
#     elif terminated:
#         reward -= 10.0
    
#     # 7. step punishment
#     reward -= 0.001 * self.current_step
    
#     # 8. alive reward
#     # reward += 0.0005
    
#     reward = np.clip(reward, -10, 10) + success_reward
    
#     return reward * 0.1


def reward_stage_1(self, old_distance, dpitch_raw, dyaw_raw, should_add_firework, terminated, truncated):
    reward = 0.0
    current_distance = np.linalg.norm(self.target_position - self.entity.position)
    
    # 1. distance reduction reward
    distance_reduction = old_distance - current_distance # -infty ~ +1.7
    
    distance_factor = 0.1 + 0.9 * (current_distance / max(self.initial_distance, 1.0)) # 0.0 ~ 1.0
    reward += distance_reduction * distance_factor * 0.5 # -infty ~ 0.85
    
    # 2. velocity orientation reward
    if np.linalg.norm(self.entity.velocity) > 0.1:
        to_target = self.target_position - self.entity.position
        to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
        velocity_dir = self.entity.velocity / (np.linalg.norm(self.entity.velocity) + 1e-6)
        direction_similarity = np.dot(velocity_dir, to_target_norm) # -1.0 ~ 1.0
        reward += direction_similarity * 0.01 # -0.01 ~ 0.01
    
    # 3. firework punishment
    if should_add_firework:
        # basic punishment
        reward -= 0.05
        # reward -= 0.1
        self.total_fireworks_used += 1
        # additional punishment
        rocket_penalty_factor = 1.0 + 0.1 * self.total_fireworks_used
        reward -= 0.05 * rocket_penalty_factor
    
    # 4. distance punishment
    # if current_distance > self.initial_distance * 1.2:  # 20%
    #     excess_factor = (current_distance - self.initial_distance * 1.2) / self.initial_distance
    #     reward -= excess_factor * 0.5
    
    # * d_pitch penalty
    reward -= 0.4 * dpitch_raw ** 2 # -0.001 ~ 0.001
    
    # * d_yaw penalty
    reward -= 10.0 * dyaw_raw ** 2     # -0.001 ~ 0.001
    
    # 5. hit
    if terminated and current_distance < 5.0:  # hit
        # fall damage
        fall_damage = self.entity.get_fall_damage()
        # clip
        capped_damage = min(fall_damage, 20.0) # 0.0 ~ 20.0
        
        # basic reward
        base_success_reward = 100.0
        
        # damage
        damage_penalty = -capped_damage * 2.0 # 0.0 ~ 40.0
        
        # firework efficiency
        rocket_efficiency_bonus = max(0, 30.0 - 5.0 * self.total_fireworks_used)
        
        # time efficiency
        time_efficiency_bonus = max(0, 20.0 - 0.2 * self.current_step)
        
        # totall reward
        success_reward = (
            base_success_reward + 
            damage_penalty + 
            rocket_efficiency_bonus + 
            time_efficiency_bonus
        )
        
        # clip
        success_reward = max(success_reward, 1.0)
        
        # info
        self._last_success_info = {
            "distance": current_distance,
            "fall_damage": fall_damage,
            "capped_damage": capped_damage,
            "rockets_used": self.total_fireworks_used,
            "steps": self.current_step,
            "time": self.total_time,
            "base_reward": base_success_reward,
            "damage_penalty": damage_penalty,
            "rocket_bonus": rocket_efficiency_bonus,
            "time_bonus": time_efficiency_bonus,
            "total_success_reward": success_reward
        }
        
        terminate_reward = success_reward
    # 6. oob punishment
    elif terminated:
        terminate_reward = -50.0
    else:
        terminate_reward = 0.0
    
    # 7. step punishment
    reward -= 0.0002 * self.current_step
    
    # 8. alive reward
    # reward += 0.0005
    
    reward = np.clip(reward, -10, 10) + terminate_reward
    
    return reward


def reward_stage_2(self, old_distance, dpitch_raw, dyaw_raw, should_add_firework, terminated, truncated):
    reward = 0.0
    current_distance = np.linalg.norm(self.target_position - self.entity.position)
    
    # 1. distance reduction reward
    # distance_reduction = old_distance - current_distance # -infty ~ +1.7
    
    # distance_factor = 0.1 + 0.9 * (current_distance / max(self.initial_distance, 1.0)) # 0.0 ~ 1.0
    # reward += distance_reduction * distance_factor * 0.5 # -infty ~ 0.85
    
    # 2. velocity orientation reward
    # if np.linalg.norm(self.entity.velocity) > 0.1:
    #     to_target = self.target_position - self.entity.position
    #     to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
    #     velocity_dir = self.entity.velocity / (np.linalg.norm(self.entity.velocity) + 1e-6)
    #     direction_similarity = np.dot(velocity_dir, to_target_norm) # -1.0 ~ 1.0
    #     reward += direction_similarity * 0.01 # -0.01 ~ 0.01
    
    # 3. firework punishment
    if should_add_firework:
        # basic punishment
        reward -= 0.05
        self.total_fireworks_used += 1
        # additional punishment
        rocket_penalty_factor = 1.0 + 0.1 * self.total_fireworks_used
        reward -= 0.05 * rocket_penalty_factor
    
    # 4. distance punishment
    # if current_distance > self.initial_distance * 1.2:  # 20%
    #     excess_factor = (current_distance - self.initial_distance * 1.2) / self.initial_distance
    #     reward -= excess_factor * 0.5
    
    # * d_pitch penalty
    # reward -= 0.01 * dpitch_raw ** 2 # -0.0025 ~ 0.0025
    
    # * d_yaw penalty
    # reward -= 0.25 * dyaw_raw ** 2   # -0.0025 ~ 0.0025
    
    # 5. hit
    success_reward = 0.0
    if terminated:  # hit
        # fall damage
        fall_damage = self.entity.get_fall_damage()
        # clip
        capped_damage = min(fall_damage, 20.0) # 0.0 ~ 20.0
        
        # basic reward
        base_success_reward = 500.0
        
        # damage
        damage_penalty = -capped_damage * 5.0 # 0.0 ~ 100.0
        
        # firework efficiency
        rocket_efficiency_bonus = max(0, 30.0 - 5.0 * self.total_fireworks_used)
        
        # time efficiency
        time_efficiency_bonus = max(0, 20.0 - 0.2 * self.current_step)
        
        # totall reward
        success_reward = (
            base_success_reward + 
            damage_penalty + 
            rocket_efficiency_bonus + 
            time_efficiency_bonus
        )
        
        # clip
        success_reward = max(success_reward, 1.0)
        
        # info
        self._last_success_info = {
            "distance": current_distance,
            "fall_damage": fall_damage,
            "capped_damage": capped_damage,
            "rockets_used": self.total_fireworks_used,
            "steps": self.current_step,
            "time": self.total_time,
            "base_reward": base_success_reward,
            "damage_penalty": damage_penalty,
            "rocket_bonus": rocket_efficiency_bonus,
            "time_bonus": time_efficiency_bonus,
            "total_success_reward": success_reward
        }
    
    # 6. oob punishment
    elif terminated:
        reward -= 10.0
    
    # 7. step punishment
    reward -= 0.01 * self.current_step
    
    # 8. alive reward
    # reward += 0.0005
    
    reward = np.clip(reward, -10, 10) + success_reward
    
    return reward * 0.1


def reward_stage_fin(self, old_distance, dpitch_raw, dyaw_raw, should_add_firework, terminated, truncated):
    reward = 0.0
    current_distance = np.linalg.norm(self.target_position - self.entity.position)
    
    # 1. distance reduction reward
    # distance_reduction = old_distance - current_distance # -infty ~ +1.7
    
    # distance_factor = 0.1 + 0.9 * (current_distance / max(self.initial_distance, 1.0)) # 0.0 ~ 1.0
    # reward += distance_reduction * distance_factor * 0.5 # -infty ~ 0.85
    
    # 2. velocity orientation reward
    # if np.linalg.norm(self.entity.velocity) > 0.1:
    #     to_target = self.target_position - self.entity.position
    #     to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
    #     velocity_dir = self.entity.velocity / (np.linalg.norm(self.entity.velocity) + 1e-6)
    #     direction_similarity = np.dot(velocity_dir, to_target_norm) # -1.0 ~ 1.0
    #     reward += direction_similarity * 0.01 # -0.01 ~ 0.01
    
    # 3. firework punishment
    if should_add_firework:
        # basic punishment
        reward -= 0.1
        self.total_fireworks_used += 1
        # additional punishment
        rocket_penalty_factor = 1.0 + 0.1 * self.total_fireworks_used
        reward -= 0.1 * rocket_penalty_factor
    
    # 4. distance punishment
    # if current_distance > self.initial_distance * 1.2:  # 20%
    #     excess_factor = (current_distance - self.initial_distance * 1.2) / self.initial_distance
    #     reward -= excess_factor * 0.5
    
    # * d_pitch penalty
    # reward -= 0.01 * dpitch_raw ** 2 # -0.0025 ~ 0.0025
    
    # * d_yaw penalty
    # reward -= 0.25 * dyaw_raw ** 2   # -0.0025 ~ 0.0025
    
    # 5. hit
    terminate_reward = 0.0
    if terminated:  # hit
        # fall damage
        fall_damage = self.entity.get_fall_damage()
        # clip
        capped_damage = min(fall_damage, 20.0) # 0.0 ~ 20.0
        
        # basic reward
        base_success_reward = self.initial_distance * 10.0
        
        # damage
        damage_penalty = -capped_damage * 5.0 # 0.0 ~ 100.0
        
        # firework efficiency
        rocket_efficiency_bonus = max(0, 30.0 - 5.0 * self.total_fireworks_used)
        
        # time efficiency
        time_efficiency_bonus = max(0, 20.0 - 0.2 * self.current_step)
        
        # totall reward
        success_reward = (
            base_success_reward + 
            damage_penalty + 
            rocket_efficiency_bonus + 
            time_efficiency_bonus * 5
        )
        
        # clip
        success_reward = max(success_reward, 1.0)
        
        # info
        self._last_success_info = {
            "distance": current_distance,
            "fall_damage": fall_damage,
            "capped_damage": capped_damage,
            "rockets_used": self.total_fireworks_used,
            "steps": self.current_step,
            "time": self.total_time,
            "base_reward": base_success_reward,
            "damage_penalty": damage_penalty,
            "rocket_bonus": rocket_efficiency_bonus,
            "time_bonus": time_efficiency_bonus,
            "total_success_reward": success_reward
        }
        
        terminate_reward = success_reward
    
    # 6. oob punishment
    elif terminated:
        terminate_reward = -10
    
    # 7. step punishment
    # reward -= 0.01 * self.current_step
    
    # 8. alive reward
    # reward += 0.0005
    
    reward = np.clip(reward, -10, 10) + terminate_reward
    
    return reward * 0.1