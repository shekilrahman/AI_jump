# enhanced_jump_game.py
import pygame as pg
import random
import math
import numpy as np
from simple_nn import SimpleNN
import os

pg.init()
pg.mixer.init()

# Screen dimensions and ground level
WIDTH, HEIGHT = 800, 500
GROUND_Y = 430

# Player dimensions
PLAYER_WIDTH, PLAYER_HEIGHT = 20, 20

# Fixed gap size
gap_size = 5 * PLAYER_HEIGHT

screen = pg.display.set_mode((WIDTH, HEIGHT))
clock = pg.time.Clock()
font = pg.font.Font(None, 30)
small_font = pg.font.Font(None, 20)

# Screen shake variables
shake_intensity = 0
shake_duration = 0
shake_offset_x = 0
shake_offset_y = 0

# Sound effects (create simple tones if files don't exist)
def create_sound_effects():
    """Create simple sound effects using pygame's sound generation"""
    sounds = {}
    
    # Jump sound - quick rising tone
    jump_samples = []
    for i in range(4410):  # 0.1 seconds at 44100 Hz
        frequency = 300 + (i / 4410) * 200  # 300Hz to 500Hz
        sample = int(16000 * math.sin(frequency * 2 * math.pi * i / 44100))
        jump_samples.append([sample, sample])
    sounds['jump'] = pg.sndarray.make_sound(np.array(jump_samples, dtype=np.int16))
    
    # Success sound - pleasant chord
    success_samples = []
    for i in range(8820):  # 0.2 seconds
        freq1, freq2, freq3 = 523, 659, 784  # C, E, G chord
        sample = int(8000 * (math.sin(freq1 * 2 * math.pi * i / 44100) +
                           math.sin(freq2 * 2 * math.pi * i / 44100) +
                           math.sin(freq3 * 2 * math.pi * i / 44100)) / 3)
        envelope = max(0, 1 - i / 8820)  # Fade out
        sample = int(sample * envelope)
        success_samples.append([sample, sample])
    sounds['success'] = pg.sndarray.make_sound(np.array(success_samples, dtype=np.int16))
    
    # Collision sound - harsh buzz
    collision_samples = []
    for i in range(6615):  # 0.15 seconds
        frequency = 150 + random.randint(-50, 50)  # Noisy low frequency
        sample = int(12000 * math.sin(frequency * 2 * math.pi * i / 44100))
        if i % 100 < 50:  # Add distortion
            sample = int(sample * 0.7)
        collision_samples.append([sample, sample])
    sounds['collision'] = pg.sndarray.make_sound(np.array(collision_samples, dtype=np.int16))
    
    # Scan sound - quick beep
    scan_samples = []
    for i in range(2205):  # 0.05 seconds
        frequency = 800
        sample = int(8000 * math.sin(frequency * 2 * math.pi * i / 44100))
        envelope = max(0, 1 - i / 2205)
        sample = int(sample * envelope)
        scan_samples.append([sample, sample])
    sounds['scan'] = pg.sndarray.make_sound(np.array(scan_samples, dtype=np.int16))
    
    return sounds

# Load or create sound effects
try:
    sounds = {
        'jump': pg.mixer.Sound('jump.wav'),
        'success': pg.mixer.Sound('success.wav'),
        'collision': pg.mixer.Sound('collision.wav'),
        'scan': pg.mixer.Sound('scan.wav')
    }
except:
    sounds = create_sound_effects()

    
import math

class DynamicNN(SimpleNN):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, initial_lr=0.1):
        super().__init__(input_size, hidden_size1, hidden_size2, output_size, initial_lr)
        self.min_lr = 0.001
        self.max_lr = 0.5
        self.prev_success_rate = 0.0
        self.last_lr_check = 0
        self.training_count = 0

    def adjust_lr(self, increase=True, factor=1.1):
        if increase:
            self.lr *= factor
        else:
            self.lr /= factor
        self.lr = max(self.min_lr, min(self.max_lr, self.lr))
        print(f"[LR ADJUST] {'→' if increase else '←'} New LR: {self.lr:.6f}")


    def update_learning_rate(self, success_rate, training_count, last_10_results):
        self.training_count = training_count

        # Reduce LR by 75% every 30 training counts (i.e., multiply by 0.25)
        if training_count > 0 and training_count % 30 == 0 and training_count != self.last_lr_check:
            self.last_lr_check = training_count
            self.lr *= 0.99
            self.lr = max(self.min_lr, min(self.max_lr, self.lr))  # Clamp within range
            print(f"[LR DROP] Count: {training_count} | LR reduced to {self.lr:.6f}")


# Initialize dynamic neural network
nn = DynamicNN(input_size=2, hidden_size1=10, hidden_size2=5, output_size=1, initial_lr=0.1)
nn_filepath = "jump_nn_weights.npz"

def apply_screen_shake(intensity):
    """Apply screen shake effect"""
    global shake_intensity, shake_duration, shake_offset_x, shake_offset_y
    shake_intensity = intensity
    shake_duration = 20  # frames
    
def update_screen_shake():
    """Update screen shake offsets"""
    global shake_intensity, shake_duration, shake_offset_x, shake_offset_y
    
    if shake_duration > 0:
        shake_offset_x = random.randint(-shake_intensity, shake_intensity)
        shake_offset_y = random.randint(-shake_intensity, shake_intensity)
        shake_duration -= 1
        shake_intensity = max(0, shake_intensity - 1)
    else:
        shake_offset_x = 0
        shake_offset_y = 0

def normalize_dist(d):
    """Normalize a Euclidean distance d into [0,1], assuming maximum ~500 px."""
    if d is None:
        return 0.0
    return min(max(d / 500.0, 0.0), 1.0)

def segment_intersection_point(x1, y1, x2, y2, x3, y3, x4, y4):
    """Compute intersection (x,y) between two line segments."""
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        xi = x1 + ua * (x2 - x1)
        yi = y1 + ua * (y2 - y1)
        return (xi, yi)
    return None

def line_rect_intersection_point(x1, y1, x2, y2, rect):
    """Return the first intersection point of line segment with rectangle."""
    edges = [
        ((rect.left, rect.top), (rect.right, rect.top)),
        ((rect.right, rect.top), (rect.right, rect.bottom)),
        ((rect.right, rect.bottom), (rect.left, rect.bottom)),
        ((rect.left, rect.bottom), (rect.left, rect.top))
    ]
    for (x3, y3), (x4, y4) in edges:
        pt = segment_intersection_point(x1, y1, x2, y2, x3, y3, x4, y4)
        if pt is not None:
            return pt
    return None

def check_collision_with_obstacles(player_x, player_y, obstacles):
    """Check if player collides with any obstacle."""
    player_rect = pg.Rect(player_x, player_y, PLAYER_WIDTH, PLAYER_HEIGHT)
    
    for obs_x, obs_h in obstacles:
        top_obs_bottom = GROUND_Y - obs_h - gap_size
        
        bottom_obs_rect = pg.Rect(obs_x, GROUND_Y - obs_h, 30, obs_h)
        
        if top_obs_bottom > 0:
            top_obs_rect = pg.Rect(obs_x, 0, 30, top_obs_bottom)
            if player_rect.colliderect(top_obs_rect):
                return ("TOP", (obs_x, obs_h))
        
        if player_rect.colliderect(bottom_obs_rect):
            return ("BOTTOM", (obs_x, obs_h))
    
    return (None, None)

def draw_particles(screen, particles):
    """Draw particle effects"""
    for particle in particles[:]:
        pg.draw.circle(screen, particle['color'], 
                      (int(particle['x']), int(particle['y'])), 
                      max(1, int(particle['size'])))
        
        # Update particle
        particle['x'] += particle['vx']
        particle['y'] += particle['vy']
        particle['vy'] += 0.1  # Gravity
        particle['size'] *= 0.98  # Shrink
        particle['life'] -= 1
        
        if particle['life'] <= 0 or particle['size'] < 0.5:
            particles.remove(particle)

def create_success_particles(x, y):
    """Create success particle effect"""
    particles = []
    for _ in range(15):
        particles.append({
            'x': x + random.randint(-10, 10),
            'y': y + random.randint(-10, 10),
            'vx': random.uniform(-3, 3),
            'vy': random.uniform(-5, -1),
            'size': random.uniform(2, 5),
            'color': (random.randint(100, 255), random.randint(200, 255), random.randint(100, 255)),
            'life': 60
        })
    return particles

def create_collision_particles(x, y):
    """Create collision particle effect"""
    particles = []
    for _ in range(20):
        particles.append({
            'x': x + random.randint(-5, 5),
            'y': y + random.randint(-5, 5),
            'vx': random.uniform(-4, 4),
            'vy': random.uniform(-6, -2),
            'size': random.uniform(1, 4),
            'color': (255, random.randint(50, 150), 0),
            'life': 45
        })
    return particles

def run():
    global shake_offset_x, shake_offset_y
    
    # Player state
    player_x = 200
    player_y = GROUND_Y - PLAYER_HEIGHT
    velocity_y = 0.0
    jumping = False

    # Game state
    obstacles = [[WIDTH, random.randint(50, 200)]]
    score = 0

    # Scanning state
    scan_angle = -90.0
    rotate_vertical = False
    rotation_stopped = False

    horizontal_length = 350
    vertical_length = 2000

    red_touch_point = None
    blue_touch_point = None

    red_distance_measured = False
    red_dist = None
    blue_dist = None

    jump_force_applied = 0.0
    jump_trigger_threshold = 0.1
    max_jump_velocity = 30.0

    # Training state
    ready_to_scan = False
    last_red_dist = None
    last_blue_dist = None
    jump_result = None
    training_data_count = 0
    success_count = 0
    success_factor = 0
    last_10_results = ["F"] * 10

    eval_obs = None
    eval_recorded = False
    collision_occurred = False
    
    # Visual effects
    particles = []
    result_display_timer = 0
    scan_sound_played = False

    while True:
        # Event Handling
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_RIGHT:
                    nn.adjust_lr(increase=True)   # Increase learning rate
                elif event.key == pg.K_LEFT:
                    nn.adjust_lr(increase=False)
                elif event.key == pg.K_r:
                    run()
                    return
                elif event.key == pg.K_s:
                    nn.save(nn_filepath)
                    print(f"Saved NN weights. Current LR: {nn.lr:.6f}")
                elif event.key == pg.K_l:
                    try:
                        nn.load(nn_filepath)
                        print(f"Loaded NN weights. Current LR: {nn.lr:.6f}")
                    except FileNotFoundError:
                        print(f"No file '{nn_filepath}' to load.")

        # Update screen shake
        update_screen_shake()

        # Apply gravity if jumping
        if jumping:
            velocity_y += 0.5
            player_y += velocity_y
            
            if not collision_occurred:
                collision_type, collided_obs = check_collision_with_obstacles(player_x, player_y, obstacles)
                
                if collision_type is not None:
                    collision_occurred = True
                    sounds['collision'].play()
                    apply_screen_shake(8)
                    particles.extend(create_collision_particles(player_x + PLAYER_WIDTH//2, 
                                                              player_y + PLAYER_HEIGHT//2))
                    
                    if collision_type == "TOP":
                        jump_result = "TOO HIGH"
                        label = 0.0
                    elif collision_type == "BOTTOM":
                        jump_result = "TOO LOW"
                        label = 1.0
                    
                    result_display_timer = 120  # Show result for 2 seconds
                    
                    if eval_obs is not None and last_red_dist is not None and last_blue_dist is not None:
                        input_vec = np.array([[normalize_dist(last_red_dist)],
                                               [normalize_dist(last_blue_dist)]])
                        target = np.array([[label]])
                        nn.train(input_vec, target)
                        training_data_count += 1
                        eval_recorded = True
            
            # Land check
            if player_y >= GROUND_Y - PLAYER_HEIGHT:
                player_y = GROUND_Y - PLAYER_HEIGHT
                jumping = False

                if (eval_obs is not None) and (not eval_recorded):
                    jump_result = "SUCCESS"
                    success_count += 1
                    label = 0.5
                    last_10_results.append("S")
                    last_10_results.pop(0)
                    
                    sounds['success'].play()
                    particles.extend(create_success_particles(player_x + PLAYER_WIDTH//2, 
                                                            player_y + PLAYER_HEIGHT//2))
                    result_display_timer = 120

                    if last_red_dist is not None and last_blue_dist is not None:
                        input_vec = np.array([[normalize_dist(last_red_dist)],
                                               [normalize_dist(last_blue_dist)]])
                        target = np.array([[label]])
                        nn.train(input_vec, target)
                        training_data_count += 1
                else:
                    last_10_results.append("F")
                    last_10_results.pop(0)

                # Update neural network learning rate
                if training_data_count > 0:
                    current_success_rate = success_count / training_data_count
                    nn.update_learning_rate(current_success_rate, training_data_count, last_10_results)

                # Reset eval state
                eval_obs = None
                eval_recorded = False
                collision_occurred = False
                ready_to_scan = False
                last_red_dist = None
                last_blue_dist = None
                scan_sound_played = False

        # Move obstacles left
        for obs in obstacles:
            obs[0] -= 4
        obstacles = [obs for obs in obstacles if obs[0] > -50]
        if len(obstacles) == 0 or obstacles[-1][0] < 300:
            obstacles.append([WIDTH, random.randint(50, 200 + score // 10)])

        # Check if ready to scan
        if not ready_to_scan and obstacles:
            if obstacles[0][0] + 30 < 0:
                ready_to_scan = True

        # Drawing with shake offset
        screen.fill((135, 206, 235))
        
        # Apply shake to all drawing operations
        shake_x, shake_y = shake_offset_x, shake_offset_y
        
        # Ground
        pg.draw.rect(screen, (34, 139, 34), 
                    (shake_x, GROUND_Y + shake_y, WIDTH, HEIGHT - GROUND_Y))
        
        # Player
        pg.draw.rect(screen, (0, 120, 255), 
                    (player_x + shake_x, player_y + shake_y, PLAYER_WIDTH, PLAYER_HEIGHT))
        
        # Add player glow effect when jumping
        if jumping:
            for i in range(3):
                alpha_surface = pg.Surface((PLAYER_WIDTH + 4*i, PLAYER_HEIGHT + 4*i))
                alpha_surface.set_alpha(30 - i*10)
                alpha_surface.fill((0, 120, 255))
                screen.blit(alpha_surface, (player_x - 2*i + shake_x, player_y - 2*i + shake_y))

        # Draw obstacles
        for obs_x, obs_h in obstacles:
            top_obs_bottom = GROUND_Y - obs_h - gap_size
            # Bottom obstacle
            pg.draw.rect(screen, (160, 82, 45), 
                        (obs_x + shake_x, GROUND_Y - obs_h + shake_y, 30, obs_h))
            # Top obstacle
            if top_obs_bottom > 0:
                pg.draw.rect(screen, (160, 82, 45), 
                            (obs_x + shake_x, shake_y, 30, top_obs_bottom))

        # Scanning & Jump decision
        if obstacles and ready_to_scan:
            obs_x, obs_h = obstacles[0]
            first_obs_rect = pg.Rect(obs_x, GROUND_Y - obs_h, 30, obs_h)

            center_x = player_x + PLAYER_WIDTH // 2
            center_y = player_y + PLAYER_HEIGHT // 2

            # Draw horizontal red line with shake
            horiz_end = (center_x + horizontal_length, center_y)
            pg.draw.line(screen, (255, 0, 0), 
                        (center_x + shake_x, center_y + shake_y), 
                        (horiz_end[0] + shake_x, horiz_end[1] + shake_y), 2)

            if not red_distance_measured:
                pt_red = line_rect_intersection_point(center_x, center_y, horiz_end[0], horiz_end[1], first_obs_rect)
                if pt_red is not None:
                    red_touch_point = pt_red
                    red_distance_measured = False

            if red_touch_point is not None:
                pg.draw.circle(screen, (255, 0, 0),
                            (int(red_touch_point[0] + shake_x), int(red_touch_point[1] + shake_y)), 6)

            # Vertical line rotation
            if not rotate_vertical:
                horiz_line_rect = pg.Rect(center_x, center_y, horizontal_length, 1)
                if horiz_line_rect.colliderect(first_obs_rect):
                    rotate_vertical = True
                    scan_angle = -90
                    if not scan_sound_played:
                        sounds['scan'].play()
                        scan_sound_played = True

            if rotate_vertical:
                if not rotation_stopped:
                    scan_angle += 2
                    if scan_angle > 90:
                        scan_angle = 90
                        rotation_stopped = True

                rad = math.radians(scan_angle)
                vert_end_x = center_x + vertical_length * math.cos(rad)
                vert_end_y = center_y + vertical_length * math.sin(rad)

                pg.draw.line(screen, (0, 0, 255),
                            (center_x + shake_x, center_y + shake_y), 
                            (vert_end_x + shake_x, vert_end_y + shake_y), 2)

                if not rotation_stopped:
                    pt_blue = line_rect_intersection_point(center_x, center_y, vert_end_x, vert_end_y, first_obs_rect)
                    if pt_blue is not None:
                        blue_touch_point = pt_blue
                        rotation_stopped = True

                        if red_touch_point is None:
                            pt_red = line_rect_intersection_point(center_x, center_y, center_x + horizontal_length, center_y, first_obs_rect)
                            if pt_red is not None:
                                red_touch_point = pt_red

                        if red_touch_point is not None:
                            red_dist = math.dist((center_x, center_y), red_touch_point)
                        else:
                            red_dist = None
                        blue_dist = math.dist((center_x, center_y), blue_touch_point)

                        input_vec = np.array([[normalize_dist(red_dist)], [normalize_dist(blue_dist)]])
                        output = nn.forward(input_vec)
                        jump_force = output[0, 0]
                        jump_force_applied = jump_force

                        scan_angle = -90
                        rotation_stopped = False
                        rotate_vertical = False
                        red_touch_point = None
                        blue_touch_point = None
                        red_distance_measured = False

                        if (not jumping) and (jump_force > jump_trigger_threshold):
                            velocity_y = -jump_force * max_jump_velocity
                            jumping = True
                            ready_to_scan = False
                            last_red_dist = red_dist
                            last_blue_dist = blue_dist
                            
                            sounds['jump'].play()
                            apply_screen_shake(3)

                            eval_obs = [obs_x, obs_h]
                            eval_recorded = False
                            collision_occurred = False

            if blue_touch_point is not None:
                pg.draw.circle(screen, (0, 0, 255),
                               (int(blue_touch_point[0] + shake_x), int(blue_touch_point[1] + shake_y)), 6)

        # Passing check for successful jumps
        if jumping and (eval_obs is not None) and (not eval_recorded) and (not collision_occurred):
            obs_x, obs_h = eval_obs
            if (player_x + PLAYER_WIDTH) > (obs_x + 30):
                top_obs_bottom = GROUND_Y - obs_h - gap_size
                player_bottom = player_y + PLAYER_HEIGHT
                
                if player_y >= top_obs_bottom and player_bottom <= (GROUND_Y - obs_h):
                    jump_result = "SUCCESS"
                    label = 0.5
                elif player_y < top_obs_bottom:
                    jump_result = "TOO HIGH"
                    label = 0.0
                elif player_bottom > (GROUND_Y - obs_h):
                    jump_result = "TOO LOW" 
                    label = 1.0
                else:
                    jump_result = "SUCCESS"
                    label = 0.5

                if last_red_dist is not None and last_blue_dist is not None:
                    input_vec = np.array([[normalize_dist(last_red_dist)], [normalize_dist(last_blue_dist)]])
                    target = np.array([[label]])
                    nn.train(input_vec, target)
                    training_data_count += 1

                eval_recorded = True

        # Update score
        score += sum(1 for obs in obstacles if 145 < obs[0] < 150)

        # Draw particles
        draw_particles(screen, particles)

        # Decrease result display timer
        if result_display_timer > 0:
            result_display_timer -= 1

        # Draw UI
        RIGHT_MARGIN = 20
        LEFT_MARGIN = 20

        def render_right(text, y, color):
            text_surface = font.render(text, True, color)
            screen.blit(text_surface, (WIDTH - text_surface.get_width() - RIGHT_MARGIN, y))

        def render_left(text, y, color):
            text_surface = font.render(text, True, color)
            screen.blit(text_surface, (LEFT_MARGIN, y))

        def render_small_left(text, y, color):
            text_surface = small_font.render(text, True, color)
            screen.blit(text_surface, (LEFT_MARGIN, y))

        if red_dist is not None:
            render_right(f"Red Dist: {red_dist:.1f}cm", 50, (255, 0, 0))
        if blue_dist is not None:
            render_right(f"Blue Dist: {blue_dist:.1f}cm", 80, (0, 0, 255))
        
        if (training_data_count-success_count) != 0:
            success_rate = (success_count/(training_data_count-success_count))*100
            render_left(f"Success Rate: {success_rate:.1f}%", 110, (0, 0, 0))
            render_right(f"Jump Force: {jump_force_applied*100:.2f}J", 110, (0, 0, 0))
            render_left(f"Learning Rate: {nn.lr:.6f}", 140, (128, 128, 128))
            
            # Draw last 10 results
            for i, result in enumerate(reversed(last_10_results)):
                color = (0, 255, 0) if result == "S" else (255, 0, 0)
                pg.draw.rect(screen, color, (500 - i * 25, 10, 20, 20))

        # Show jump result with timer
        if result_display_timer > 0:
            if jump_result == "SUCCESS":
                render_right("SUCCESS!", 20, (0, 200, 0))
            elif jump_result == "TOO HIGH":
                render_right("TOO HIGH!", 20, (255, 100, 0))
            elif jump_result == "TOO LOW":
                render_right("TOO LOW!", 20, (200, 0, 200))
        
        render_left(f"Training Samples: {training_data_count}", 20, (0, 0, 0))
        render_left(f"Success: {success_count}", 50, (0, 0, 0))
        render_left(f"Failed: {training_data_count - success_count}", 80, (0, 0, 0))

        # Controls hint
        render_small_left("R: Restart | S: Save | L: Load", HEIGHT - 25, (100, 100, 100))

        pg.display.flip()
        clock.tick(60)

# Entry point
if __name__ == "__main__":
    run()