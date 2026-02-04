import numpy as np
import random
import math

class LivingEntity:
    def __init__(self, position=np.array([0.0, 0.0, 0.0]), velocity=np.array([0.0, 0.0, 0.0]), rotation=np.array([0.0, 0.0])):
        self.position = position
        self.velocity = velocity
        self.rotation = rotation  # [yaw, pitch]
        self.fall_distance = 0.0
        
        
    def adjust_orientation(self, dyaw_degrees, dpitch_degrees):
        self.rotation[0] += dyaw_degrees
        self.rotation[0] = self._normalize_yaw(self.rotation[0])
        
        self.rotation[1] += dpitch_degrees
        self.rotation[1] = np.clip(self.rotation[1], -90.0, 90.0)
    
    def _normalize_yaw(self, yaw):
        normalized = yaw % 360.0
        
        if normalized < 0:
            normalized += 360.0
        
        return normalized
    
    @property
    def normalized_rotation(self):
        return np.array([
            self._normalize_yaw(self.rotation[0]),
            np.clip(self.rotation[1], -90.0, 90.0)
        ])
    
    def set_orientation(self, yaw_degrees, pitch_degrees):
        self.rotation[0] = self._normalize_yaw(yaw_degrees)
        self.rotation[1] = np.clip(pitch_degrees, -90.0, 90.0)
        
        
    def get_pitch_radians(self):
        return np.radians(self.rotation[1])
        
    def get_look_vector(self):
        yaw_rad = np.radians(self.rotation[0])
        pitch_rad = np.radians(self.rotation[1])
        
        x = -np.sin(yaw_rad) * np.cos(pitch_rad)
        y = -np.sin(pitch_rad)
        z = np.cos(yaw_rad) * np.cos(pitch_rad)
        
        return np.array([x, y, z])
        
    def calculate_horizontal_distance(self, vector):
        return np.sqrt(vector[0]**2 + vector[2]**2)

    def get_fall_damage(self, damage_multiplier=1.0):
        base_damage = self.fall_distance - 3.0
        base_damage = max(base_damage, 0)
        damage = base_damage * damage_multiplier
        
        return math.ceil(damage)
    
    def check_slow_fall_distance(self):
        if self.velocity[1] > -0.5 and self.fall_distance > 1.0:
            self.fall_distance = 1.0
        
    def get_collision_damage(self):
        damage = self.calculate_horizontal_distance(self.velocity) * 10.0 - 3.0
        damage = max(damage, 0)
            
        return damage
        
    def travel_elytra(self):
        self.check_slow_fall_distance()
        
        current_vel = self.velocity
        look_vec = self.get_look_vector()
        pitch = self.get_pitch_radians()
        
        horizontal_look = np.sqrt(look_vec[0] * look_vec[0] + look_vec[2] * look_vec[2])
        horizontal_speed = self.calculate_horizontal_distance(current_vel)
        look_length = np.linalg.norm(look_vec)
        cos_pitch = np.cos(pitch)
        cos_pitch_sq = cos_pitch * cos_pitch * min(1.0, look_length / 0.4)
        
        new_vel = current_vel + np.array([
            0.0, 
            0.08 * (-1.0 + cos_pitch_sq * 0.75), 
            0.0
        ])
        
        if new_vel[1] < 0.0 and horizontal_look > 0.0:
            lift = new_vel[1] * -0.1 * cos_pitch_sq
            new_vel += np.array([
                look_vec[0] * lift / horizontal_look,
                lift,
                look_vec[2] * lift / horizontal_look
            ])
        
        if pitch < 0.0 and horizontal_look > 0.0:
            climb = horizontal_speed * (-np.sin(pitch)) * 0.04
            new_vel += np.array([
                -look_vec[0] * climb / horizontal_look,
                climb * 3.2,
                -look_vec[2] * climb / horizontal_look
            ])
        
        if horizontal_look > 0.0:
            horizontal_adjust = np.array([
                (look_vec[0] / horizontal_look * horizontal_speed - new_vel[0]) * 0.1,
                0.0,
                (look_vec[2] / horizontal_look * horizontal_speed - new_vel[2]) * 0.1
            ])
            new_vel += horizontal_adjust
        
        self.velocity = new_vel * np.array([0.99, 0.98, 0.99])
        
        if self.velocity[1] < 0.0:
            self.fall_distance -= self.velocity[1]
        
        self.position += self.velocity
        
        return self.velocity

class FireworkRocketEntity:
    def __init__(self, attached_entity: LivingEntity, flight_level: int=0):
        # flight_levet: 0~2
        self.attached_entity = attached_entity
        self.life = 0
        self.is_exploded = False
        
        self.lifetime = 10 * (1 + flight_level) + random.randint(0, 5) + random.randint(0, 6)
        
    def tick_attached(self):
        if self.attached_entity is None:
            return
            
        look_vec = self.attached_entity.get_look_vector()
        current_vel = self.attached_entity.velocity
        
        target_vel = look_vec * 1.5
        acceleration = look_vec * 0.1 + (target_vel - current_vel) * 0.5
        new_vel = current_vel + acceleration
        
        self.attached_entity.velocity = new_vel
        
        self.position = self.attached_entity.position.copy()
        self.velocity = new_vel.copy()
        
    def tick(self):
        self.tick_attached()
        
        self.life += 1
        if self.life > self.lifetime:
            self.explode()
    
    def explode(self):
        self.is_exploded = True