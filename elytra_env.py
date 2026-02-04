import gymnasium as gym
from gymnasium import spaces

import numpy as np
from entity import LivingEntity
from firework_pool import FireworkPool

import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D

class ElytraEnv(gym.Env):
    def __init__(self, config=None):
        super(ElytraEnv, self).__init__()
        
        self.config = config or {}
        
        self.firework_threshold = self.config.get('firework_threshold', 0.5)
        
        # self.fixed_yaw = self.config.get('fixed_yaw', True)
        self.max_steps = self.config.get('max_steps', 1000)
        self.fps = 20
        self.dt = 1.0 / self.fps
        
        # 3D render option
        self.render_mode = self.config.get('render_mode', 'human')  # 'human' 或 '3d'
        self.render_interval = self.config.get('render_interval', 10)  # 每10步渲染一次
        
        # 3D render vars
        self.fig = None
        self.ax = None
        self.ani = None
        self.trajectory_line = None
        self.entity_point = None
        self.target_point = None
        self.view_init_elev = 30
        self.view_init_azim = 45
        
        # success
        self.success_distance = self.config.get('success_distance', 5.0)
        
        # ActionSpace: [
        #    dpitch,       # -0.5 ~  0.5 => -25 ~  25°
        #    dyaw,         # -0.1 ~  0.1 => -5  ~  5°
        #    firework_prob #  0.0 ~  1.0
        # ]
        self.dpitch_limit = self.config.get('dpitch_limit', 0.5)
        self.dyaw_limit = self.config.get('dyaw_limit', 0.1)
        self.action_space = spaces.Box(
            low=np.array([-self.dpitch_limit, -self.dyaw_limit, 0.0], dtype=np.float32),
            high=np.array([self.dpitch_limit, self.dyaw_limit, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        # StateSpace: [dx, dy, dz, vx, vy, vz, pitch, yaw, firework_attached]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32
        )
        
        self.entity = None
        self.firework_pool = None
        
        self.target_position = None
        
        self.current_step = 0
        self.total_time = 0.0
        
        self._init_normalization_params()
        
        self._init_target_generation_params()
        
        self._init_action_scaling_params()
        
        print(f"initialization done.")
        # print(f"initialization done: yaw={self.fixed_yaw}, max_step={self.max_steps}")
    
    def _init_3d_render(self):
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 10))
            
            self.ax_3d = self.fig.add_subplot(221, projection='3d')
            self.ax_trajectory = self.fig.add_subplot(222)
            self.ax_speed = self.fig.add_subplot(223)
            self.ax_reward = self.fig.add_subplot(224)
            
            plt.ion()
            plt.show(block=False)
            
            self.trajectory_points = []
            self.speed_history = []
            self.reward_history = []
            self.step_history = []
            
            self.ax_3d.set_xlabel('X')
            self.ax_3d.set_ylabel('Z (Horizontal)')
            self.ax_3d.set_zlabel('Y (Up)')
            self.ax_3d.set_title('3D Flight Trajectory')
            
            self.ax_trajectory.set_xlabel('X')
            self.ax_trajectory.set_ylabel('Z')
            self.ax_trajectory.set_title('Top View (X-Z)')
            self.ax_trajectory.grid(True)
            
            self.ax_speed.set_xlabel('Step')
            self.ax_speed.set_ylabel('Speed (m/s)')
            self.ax_reward.set_xlabel('Step')
            self.ax_reward.set_ylabel('Reward')
            
            plt.tight_layout()
    
    def _reset_3d_render(self):
        self.trajectory_points = []
        self.speed_history = []
        self.reward_history = []
        self.step_history = []
        self.pitch_history = []
        self.yaw_history = []
        
    def _update_3d_render(self):
        if self.entity is None or self.target_position is None:
            return
        
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            return
        
        self.trajectory_points.append(self.entity.position.copy())
        current_speed = np.linalg.norm(self.entity.velocity)
        self.speed_history.append(current_speed)
        self.step_history.append(self.current_step)
        
        self.ax_3d.clear()
        self.ax_trajectory.clear()
        self.ax_speed.clear()
        self.ax_reward.clear()
        
        if len(self.trajectory_points) > 1:
            trajectory = np.array(self.trajectory_points)
            
            self.ax_3d.plot(trajectory[:, 0], trajectory[:, 2], trajectory[:, 1], 
                        'b-', alpha=0.6, linewidth=1, label='Trajectory')
            self.ax_3d.scatter(self.entity.position[0], self.entity.position[2], self.entity.position[1],
                            c='r', s=100, marker='o', label='Entity')
            self.ax_3d.scatter(self.target_position[0], self.target_position[2], self.target_position[1],
                            c='g', s=200, marker='*', label='Target')
            
            arrow_length = 10.0
            v = self.entity.velocity / (np.linalg.norm(self.entity.velocity) + 1e-6)
            self.ax_3d.quiver(self.entity.position[0], self.entity.position[2], self.entity.position[1],
                            v[0]*arrow_length, v[2]*arrow_length, v[1]*arrow_length,
                            color='r', label='Velocity')
            
            look = self.entity.get_look_vector()
            self.ax_3d.quiver(self.entity.position[0], self.entity.position[2], self.entity.position[1],
                            look[0]*arrow_length, look[2]*arrow_length, look[1]*arrow_length,
                            color='orange', label='Facing')
            
            self.ax_3d.set_xlabel('X')
            self.ax_3d.set_ylabel('Z')
            self.ax_3d.set_zlabel('Y')
            self.ax_3d.view_init(elev=self.view_init_elev, azim=self.view_init_azim)
            
            all_pts = np.vstack([trajectory, [self.target_position]])
            min_v = all_pts.min(axis=0) - 20
            max_v = all_pts.max(axis=0) + 20
            self.ax_3d.set_xlim(min_v[0], max_v[0])
            self.ax_3d.set_ylim(min_v[2], max_v[2])
            self.ax_3d.set_zlim(min_v[1], max_v[1])

            self.ax_trajectory.plot(trajectory[:, 0], trajectory[:, 2], 'b-')
            self.ax_trajectory.scatter(self.target_position[0], self.target_position[2], c='g', marker='*')
            self.ax_trajectory.set_xlabel('X')
            self.ax_trajectory.set_ylabel('Z')
            self.ax_trajectory.axis('equal')
            
            if len(self.speed_history) > 0:
                self.ax_speed.plot(self.step_history, self.speed_history, 'g-', linewidth=2)
                self.ax_speed.set_xlabel('Step')
                self.ax_speed.set_ylabel('Speed (m/s)')
                self.ax_speed.set_title(f'Speed: {current_speed:.1f} m/s')
                self.ax_speed.grid(True)
                self.ax_speed.set_ylim(0, max(max(self.speed_history) * 1.1, 5.0))
            
            if hasattr(self, 'reward_history') and len(self.reward_history) > 0:
                self.ax_reward.plot(range(len(self.reward_history)), self.reward_history, 'r-', linewidth=2)
                self.ax_reward.set_xlabel('Step')
                self.ax_reward.set_ylabel('Reward')
                self.ax_reward.set_title(f'Current Reward: {self.reward_history[-1]:.2f}')
                self.ax_reward.grid(True)
        
        plt.draw()
        plt.pause(0.02)
    
    def _init_normalization_params(self):
        self.pos_normalization_method = 'relative' # 'relative', 'log', 'tanh'
        self.pos_scale = self.config.get('pos_scale', 200.0)  # scale for 'tanh'
        
        self.vel_scale = self.config.get('vel_scale', 50.0)
        
        self.angle_scale = self.config.get('angle_scale', 100.0)
    
    def _init_target_generation_params(self):
        self.min_target_distance = self.config.get('min_target_distance', 100.0)
        self.max_target_distance = self.config.get('max_target_distance', 300.0)
        
    def _init_action_scaling_params(self):
        # dpitch: [-0.5, 0.5] -> [-25, 25] degree
        # dyaw: [-0.1, 0.1] -> [-5, 5] degree
        
        self.pitch_scale = self.config.get('pitch_scale', 50.0)
        self.yaw_scale = self.config.get('yaw_scale', 50.0)
        
        # if self.fixed_yaw:
        #     self.yaw_scale = 0.0
    
    def _generate_spherical_target(self, start_position):
        r = self.np_random.uniform(self.min_target_distance, self.max_target_distance)
        
        u = self.np_random.uniform(0, 1)
        v = self.np_random.uniform(0, 1)
        
        theta = np.acos(2 * u - 1)
        phi = 2 * np.pi * v
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.cos(theta)
        z = r * np.sin(theta) * np.sin(phi)
        
        target_position = start_position + np.array([x, y, z])
        
        return target_position
    
    def _normalize_position(self, position):
        if self.pos_normalization_method == 'tanh':
            return np.tanh(position / self.pos_scale)
        elif self.pos_normalization_method == 'log':
            return np.sign(position) * np.log1p(np.abs(position))
        elif self.pos_normalization_method == 'relative':
            norm = np.linalg.norm(position)
            if norm < 1e-6:
                return position
            return position / (norm + 1.0)
        else:
            return position
    
    def _normalize_velocity(self, velocity):
        return velocity / self.vel_scale
    
    def _normalize_angle(self, angle):
        return angle / self.angle_scale

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.render_mode == '3d':
            self._reset_3d_render()
        
        initial_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.initial_position = initial_position
        self.target_position = self._generate_spherical_target(initial_position)
        self.entity = LivingEntity(
            initial_position, 
            self.np_random.uniform(-3.0, 3.0, size=3)
        )
        
        # to_target = self.target_position - self.entity.position
        # target_direction = to_target / (np.linalg.norm(to_target) + 1e-6)
        
        # yaw = np.degrees(np.arctan2(-target_direction[0], target_direction[2]))
        # yaw = yaw % 360.0
        
        # horizontal_length = np.sqrt(target_direction[0]**2 + target_direction[2]**2)
        # pitch = np.degrees(np.arctan2(-target_direction[1], horizontal_length))
        
        u = self.np_random.uniform(0, 1)
        v = self.np_random.uniform(0, 1)
        
        pitch = np.acos(2 * u - 1) - np.pi / 2
        yaw = 2 * np.pi * v
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)
        
        assert -90.0 <= pitch <= +90.0
        assert 0.0 <= yaw <= +360
        
        # pitch = np.clip(pitch, -10.0, 15.0)
        
        self.entity.set_orientation(yaw, pitch)
        
        # initial_speed = 0.0  # 5 m/s
        # look_vec = self.entity.get_look_vector()
        # self.entity.velocity = look_vec * initial_speed
        
        self.firework_pool = FireworkPool(self.entity)
        
        self.current_step = 0
        self.total_time = 0.0
        
        self.previous_distance = np.linalg.norm(self.target_position - self.entity.position)
        self.initial_distance = self.previous_distance
        self._last_firework_count = 0
        self.total_fireworks_used = 0
        
        if hasattr(self, 'trajectory_points'):
            self.trajectory_points.clear()
        else:
            self.trajectory_points = []
        self.trajectory_points.append(self.entity.position.copy())
        
        obs = self._get_obs()
        info = {
            "position": self.entity.position.copy(),
            "target": self.target_position.copy(),
            "orientation": self.entity.normalized_rotation.copy(),
            "step": self.current_step,
            "time": self.total_time,
            "distance": self.previous_distance,
            "initial_distance": self.initial_distance,
            "velocity": self.entity.velocity.copy(),
            "initial_yaw": yaw,
            "initial_pitch": pitch
        }
        
        return obs, info

    def _get_obs(self):
        if self.entity is None:
            return np.zeros(9, dtype=np.float32)
        
        # 1. get reletive position
        if self.target_position is None:
            relative_pos = np.zeros(3, dtype=np.float32)
        else:
            relative_pos = self.target_position - self.entity.position
        
        # 2. get current velocity
        velocity = self.entity.velocity
        
        # 3. get current rotation (pitch, yaw)
        yaw, pitch = self.entity.rotation
        
        # 4. firework state
        if self.firework_pool is not None and self.firework_pool.has_active_firework():
            firework_attached = 1.0
        else:
            firework_attached = 0.0
        
        # 5. normalization
        normalized_pos =   self._normalize_position(relative_pos)
        normalized_vel =   self._normalize_velocity(velocity)
        normalized_pitch = self._normalize_angle(pitch)
        normalized_yaw =   self._normalize_angle(yaw)
        
        # 6. combine
        obs = np.array([
            *normalized_pos, 
            *normalized_vel, 
            normalized_pitch, normalized_yaw, 
            firework_attached 
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        if self.entity is None:
            raise NotImplementedError('`self.entity` is `None`')
        if self.firework_pool is None:
            raise NotImplementedError('`self.firework_pool` is `None`')
        if self.target_position is None:
            raise NotImplementedError('`self.target_position` is `None`')
        
        dpitch_raw, dyaw_raw, firework_prob = action
        
        # 1. apply scale
        dpitch_degrees = dpitch_raw * self.pitch_scale
        # if self.fixed_yaw:
        #     dyaw_degrees = 0.0
        # else:
        #     dyaw_degrees = dyaw_raw * self.yaw_scale
        dyaw_degrees = dyaw_raw * self.yaw_scale
        
        # 2. adjust orientation
        self.entity.adjust_orientation(dyaw_degrees, dpitch_degrees)
        
        # 3. firework        
        should_add_firework = firework_prob > self.firework_threshold
        
        if should_add_firework:
            self.firework_pool.add_firework()
        
        # 4. travel
        self.entity.travel_elytra()
        
        # 5. tick `firework_pool`
        self.firework_pool.tick()
        
        # 6. update step & time
        self.current_step += 1
        self.total_time += self.dt
        
        # 7. check if terminated / truncated
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        # 8. calc reward
        old_distance = np.linalg.norm(self.target_position - self.entity.position)
        reward = self._calculate_reward(
            old_distance, 
            dpitch_raw, 
            dyaw_raw, 
            should_add_firework, 
            terminated, 
            truncated, 
        )
        
        # 9. get new obs
        obs = self._get_obs()
        
        # 10. build info dict
        info = {
            "position": self.entity.position.copy(),
            "velocity": self.entity.velocity.copy(),
            "orientation": self.entity.normalized_rotation.copy(),
            "target_position": self.target_position.copy(),
            "distance_to_target": np.linalg.norm(self.target_position - self.entity.position),
            "firework_active": self.firework_pool.has_active_firework(),
            "firework_count": len(self.firework_pool),
            "step": self.current_step,
            "time": self.total_time,
            "action_taken": action.copy(),
            "dpitch_degrees": dpitch_degrees,
            "dyaw_degrees": dyaw_degrees,
            "added_firework": should_add_firework, 
            "episode_success_reason": "in_progress",
        }
        
        # 11. terminate reason
        if terminated:
            info["termination_reason"] = self._get_termination_reason()
            episode_success = old_distance < self.success_distance
            info['episode_success'] = episode_success
            info['episode_success_reason'] = "success" if episode_success else "out_of_bounds"
            
        if truncated:
            info["truncation_reason"] = "max_steps_reached"
            info['episode_success'] = False
            info['episode_success_reason'] = "max_steps_reached"
            
        # 12. render if needed
        if self.render_mode == '3d':
            self.render(mode='3d', reward=reward)
        elif self.render_mode == 'human':
            self.render(mode='human')
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        self.entity = None
        if self.firework_pool:
            self.firework_pool.clear()
        self.firework_pool = None
        
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
            self.fig = None
        
        self.target_position = None
        self.current_step = 0
        self.total_time = 0.0
    
    def _check_terminated(self):
        if self.entity is None:
            raise NotImplementedError('`self.entity` is `None`')
        
        distance = np.linalg.norm(self.target_position - self.entity.position)
        # if distance < 5.0:
        if distance < self.success_distance:
            return True
        
        max_distance = 1000.0
        if np.linalg.norm(self.entity.position) > max_distance:
            return True
        
        return False

    def _check_truncated(self):
        return self.current_step >= self.max_steps

    def _calculate_reward(self, old_distance, dpitch_raw, dyaw_raw, should_add_firework, terminated, truncated):
        from reward import reward_stage_fin as _calculate_reward
        return _calculate_reward(
            self, 
            old_distance, 
            dpitch_raw, 
            dyaw_raw,  
            should_add_firework, 
            terminated, 
            truncated,
        )

    def _get_termination_reason(self):
        if self.entity is None:
            raise NotImplementedError('`self.entity` is `None`')
        
        distance = np.linalg.norm(self.target_position - self.entity.position)
        
        if distance < self.success_distance:
            fall_damage = self.entity.get_fall_damage()
            capped_damage = min(fall_damage, 20.0)
            rockets_used = self.total_fireworks_used if hasattr(self, 'total_fireworks_used') else 0
            return f"success: damage={fall_damage:.1f}(capped:{capped_damage:.1f}), rockets={rockets_used}, steps={self.current_step}"
        elif np.linalg.norm(self.entity.position) > 1000.0:
            return "out_of_bounds"
        elif self.current_step >= self.max_steps:
            return "max_steps"
        else:
            return "unknown"
    
    def _render_text(self):
        if self.entity is None:
            print("please call `reset()` first.")
            return
        
        distance = np.linalg.norm(self.target_position - self.entity.position)
        speed = np.linalg.norm(self.entity.velocity)
        
        print(f"Step: {self.current_step}/{self.max_steps}, "
              f"Dist: {distance:.1f}m, "
              f"Speed: {speed:.1f}m/s, "
              f"Pos: ({self.entity.position[0]:.0f}, {self.entity.position[1]:.0f}, {self.entity.position[2]:.0f}), "
              f"Rockets: {self.total_fireworks_used}")
    
    def render(self, mode=None, reward=None):
        if mode is None:
            mode = self.render_mode
        
        if mode == 'human':
            self._render_text()
        elif mode == '3d':
            if self.fig is None:
                self._init_3d_render()
            
            if reward is not None:
                if not hasattr(self, 'reward_history'):
                    self.reward_history = []
                self.reward_history.append(reward)
            
            if self.current_step % self.render_interval == 0 or self.current_step <= 1:
                self._update_3d_render()
        elif mode == 'rgb_array':
            return np.zeros((400, 600, 3), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported mode. ")