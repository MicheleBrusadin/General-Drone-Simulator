import math
import numpy as np
import pygame

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0,255,0)
DRONE_SIZE = 40  # Size of the drone square
MOTOR_SIZE = 10   # Size of the motor squares

class Display:
    def __init__(self, config: dict, update_frequency: int, title: str):
        self.width = config["width"]
        self.height = config["height"]
        self.update_frequency = update_frequency

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def _draw_drone(self, drone):
        # Drone state
        state = drone.get_state()
        drone_x, drone_y, _, _, rotation, _ = state
        
        width = abs(min([motor[0] for motor in drone.motors]) - max([motor[0] for motor in drone.motors]))
        height = abs(min([motor[1] for motor in drone.motors]) - max([motor[1] for motor in drone.motors]))

        surface_width = max(width*100 + MOTOR_SIZE, DRONE_SIZE)
        surface_height = max(height*100 + MOTOR_SIZE, DRONE_SIZE)

        drone_surface = pygame.Surface((surface_width, surface_height), pygame.SRCALPHA)  # Use SRCALPHA for transparency

        # Draw the drone rectangle on the surface
        drone_rect = pygame.Rect(surface_width/2 - DRONE_SIZE/2, surface_height/2 - DRONE_SIZE/2, DRONE_SIZE, DRONE_SIZE)
        pygame.draw.rect(drone_surface, WHITE, drone_rect)

        # Draw the motors
        for motor in drone.motors:
            motor_x, motor_y, _ = motor

            motor_x_scaled = motor_x * 100 + surface_width/2
            motor_y_scaled = motor_y * 100 + surface_height/2

            motor_rect = pygame.Rect(motor_x_scaled - MOTOR_SIZE/2, motor_y_scaled - MOTOR_SIZE/2, MOTOR_SIZE, MOTOR_SIZE)
            pygame.draw.rect(drone_surface, RED, motor_rect)

        # Rotate the combined drone and motors surface
        rotation = rotation * 180 / math.pi
        rotated_drone_surface = pygame.transform.rotate(drone_surface, -rotation)

        # Calculate the center position for blitting
        blit_x = drone_x - rotated_drone_surface.get_width() / 2
        blit_y = drone_y - rotated_drone_surface.get_height() / 2

        # Draw the rotated drone surface on the screen
        self.screen.blit(rotated_drone_surface, (blit_x, blit_y))

    def _draw_target(self, target):
        # Draw a dot and a circle around the target with radius 50
        pygame.draw.circle(self.screen, GREEN, (target["x"], target["y"]), target["distance"], 1)
        pygame.draw.circle(self.screen, GREEN, (target["x"], target["y"]), 2)

    def _draw_state(self, state):
        font = pygame.font.SysFont(None, 24)
        y_offset = 20  # Starting y position for the first line of text
        x_offset = 20  # Starting x position for the first line of text
        line_height = 25  # Height of each line of text

        state_labels = ["x", "y", "vx", "vy", "phi", "vphi"]

        for label, value in zip(state_labels, state):
            text = font.render(f"{label}: {round(value, 2)}", True, WHITE)
            self.screen.blit(text, (x_offset, y_offset))
            y_offset += line_height

    def _draw_agent_state(self, agent):
        font = pygame.font.SysFont(None, 24)
        y_offset = 20
        x_offset = 20

        text = font.render(f"Game: {agent.n_games}", True, WHITE)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))
        
        y_offset += 25

        text = font.render(f"Epsilon: {round(agent.epsilon*100, 1)}", True, WHITE)
        self.screen.blit(text, (self.width - text.get_width() - x_offset, y_offset))
    
        
    def _draw_action(self, action):
        # Draw at the bottom left
        font = pygame.font.SysFont(None, 24)
        y_offset = self.height - 150
        line_height = 25

        text = font.render("Action:", True, WHITE)
        self.screen.blit(text, (0, y_offset))
        y_offset += line_height

        for i, value in enumerate(action):
            text = font.render(f"{i}: {round(value * 100)}", True, WHITE)
            self.screen.blit(text, (0, y_offset))
            y_offset += line_height

    # def update(self, target, drone, agent, total_reward, action):
    #     self.clock.tick(60)
    #     with self.update_lock:
    #         if pygame.time.get_ticks() - self.last_update > 1000 / self.update_frequency:
    #             self.screen.fill(BLACK)
    #             self.last_update = pygame.time.get_ticks()

    #     self._draw_drone(drone)
    #     self._draw_target(target)
    #     #self._draw_state(drone.get_normalized_state(target))
    #     self._draw_agent_state(drone, agent, total_reward)
    #     #self._draw_action(action)
    #     pygame.display.flip()


    def draw_weights(self, surface, weights, x_start, y_start, neuron_radius, layer_spacing, neuron_spacing):
        # Calculate the starting x position based on the total number of layers
        num_layers = len(weights) // 2  # Each layer has a weight and a bias matrix

        # Find the overall min and max weights for normalization across all layers
        all_weights = np.concatenate([weights[f'layers.{2 * i}.weight'].flatten() for i in range(num_layers)])
        min_weight = np.min(all_weights)
        max_weight = np.max(all_weights)

        x = x_start

        # Draw neurons and weights layer by layer
        for i in range(num_layers):
            layer_weight_key = f'layers.{2 * i}.weight'
            weight_matrix = weights[layer_weight_key]

            # Draw neurons for current layer
            y = y_start + (max(weight_matrix.shape) - weight_matrix.shape[0]) * neuron_spacing // 2
            for neuron_index in range(weight_matrix.shape[0]):
                pygame.draw.circle(surface, WHITE, (x, y + neuron_index * neuron_spacing), neuron_radius)

            # If it's the last layer, break after drawing neurons
            if i == num_layers - 1:
                break

            # Draw connections to the next layer
            next_layer_weight_key = f'layers.{2 * (i + 1)}.weight'
            next_weight_matrix = weights[next_layer_weight_key]
            next_y = y_start + (max(next_weight_matrix.shape) - next_weight_matrix.shape[1]) * neuron_spacing // 2

            for neuron_index in range(weight_matrix.shape[0]):
                for next_neuron_index in range(weight_matrix.shape[1]):
                    weight_value = weight_matrix[neuron_index][next_neuron_index]
                    normalized_weight = (weight_value - min_weight) / (max_weight - min_weight)
                    color_intensity = int(normalized_weight * 255)
                    color = (color_intensity, color_intensity, color_intensity)  # Grayscale
                    # Draw line for weight
                    pygame.draw.line(surface, color, (x, y + neuron_index * neuron_spacing),
                                    (x + layer_spacing, next_y + next_neuron_index * neuron_spacing), 1)

            x += layer_spacing

        # Draw output neurons
        output_weights_key = f'layers.{2 * (num_layers - 1)}.weight'
        output_weight_matrix = weights[output_weights_key]
        y = y_start + (max(output_weight_matrix.shape) - output_weight_matrix.shape[1]) * neuron_spacing // 2
        for neuron_index in range(output_weight_matrix.shape[1]):
            pygame.draw.circle(surface, WHITE, (x, y + neuron_index * neuron_spacing), neuron_radius)



    def _draw_model_weights(self, agent):
        weights = agent.get_model_weights()
        weights = {k: v.cpu().numpy() for k, v in weights.items()}  # Move weights to CPU and convert to numpy

        # Define the starting position
        x_start, y_start = 100, 100
        # Define the size of neurons and spacing
        neuron_radius = 20
        layer_spacing = 150
        neuron_spacing = 60

        # Draw the weights on the Pygame surface
        self.draw_weights(self.screen, weights, x_start, y_start, neuron_radius, layer_spacing, neuron_spacing)

    def update(self, target, drones, agent):
        self.clock.tick(60)
        self.screen.fill(BLACK)

        for drone in drones:
            self._draw_drone(drone)
        self._draw_target(target)
        #self._draw_state(drone.get_normalized_state(target))
        self._draw_agent_state(agent)
        #self._draw_model_weights(agent)
        pygame.display.flip()
