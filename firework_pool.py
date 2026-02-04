from entity import LivingEntity, FireworkRocketEntity


class FireworkPool:    
    def __init__(self, attached_entity: LivingEntity):
        self.attached_entity = attached_entity
        self.fireworks = []
        
    def add_firework(self):
        firework = FireworkRocketEntity(self.attached_entity)
        self.fireworks.append(firework)
    
    def tick(self):
        self.cleanup_dead_fireworks()
        
        for firework in self.fireworks:
            firework.tick()
    
    def tick_attached(self):
        self.cleanup_dead_fireworks()
        
        for firework in self.fireworks:
            firework.tick_attached()
    
    def cleanup_dead_fireworks(self):
        alive_fireworks = []
        
        for firework in self.fireworks:
            if not firework.is_exploded:
                alive_fireworks.append(firework)
        
        self.fireworks = alive_fireworks
    
    def get_active_count(self):
        return len(self.fireworks)
    
    def has_active_firework(self):
        return self.get_active_count() > 0
    
    def get_active_firework(self):
        if self.fireworks:
            return self.fireworks[0]
        return None
    
    def clear(self):
        self.fireworks = []
    
    def __len__(self):
        return len(self.fireworks)
    
    def __str__(self):
        return f"FireworkPool(active={len(self.fireworks)})"