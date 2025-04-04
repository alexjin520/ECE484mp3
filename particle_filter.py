import numpy as np
import matplotlib.pyplot as plt
from maze import Maze, Particle, Robot
import bisect
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode
import time
import random
import os
import queue

def vehicle_dynamics(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    dx = vr * np.cos(curr_theta) * t 
    dy = vr * np.sin(curr_theta) * t
    dtheta = delta * t
    return [dx,dy,dtheta]

class particleFilter:
    def __init__(self, bob: Robot, world: Maze, num_particles, sensor_limit, x_start, y_start):
        self.num_particles = num_particles
        self.sensor_limit = sensor_limit
        particles = []

        for _ in range(num_particles):
            # x = np.random.uniform(0, world.width)
            # y = np.random.uniform(0, world.height)
            x = np.random.uniform(world.width/2, world.width)
            y = np.random.uniform(world.height/2, world.height)
            heading = np.random.uniform(0, 2 * np.pi)
            particles.append(Particle(x=x, y=y, heading=heading, maze=world, sensor_limit=sensor_limit))

        self.particles = particles
        self.variance = self.getParticleVariance()
        self.bob = bob
        self.world = world
        self.x_start = x_start
        self.y_start = y_start
        self.modelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size=1)
        self.control = []
        return

    def __controlHandler(self, data):
        tmp = list(data.data)
        self.control.append(tmp)

    def getModelState(self):
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name='polaris')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
        return modelState

    def weight_gaussian_kernel(self,x1, x2, var = 5000):
        if x1 is None:
            return 1./len(self.particles)
        tmp1 = np.array(x1)
        tmp2 = np.array(x2)
        return np.sum(np.exp(-((tmp2-tmp1) ** 2) / (2 * var)))

    def updateWeight(self, readings_robot):
        weights = []
        for p in self.particles:
            measurements_particle = p.read_sensor()
            sensvar = 5000 if self.variance > 150 else 900
            w = self.weight_gaussian_kernel(readings_robot, measurements_particle, var=sensvar)
            weights.append(w)

        weights = np.array(weights)
        total = np.sum(weights)

        # if no particles match, reinitialize around bob
        if np.max(weights) < 1e-6 or total == 0:
            print("Low particle match: resetting particle cloud around Bob.")
            self.particles = []
            for _ in range(self.num_particles):
                x = np.random.uniform(self.bob.x - 10, self.bob.x + 10)
                y = np.random.uniform(self.bob.y - 10, self.bob.y + 10)
                heading = np.random.uniform(0, 2 * np.pi)
                self.particles.append(Particle(x=x, y=y, heading=heading, maze=self.world, sensor_limit=self.sensor_limit))
            return

        weights = weights / total
        for i, p in enumerate(self.particles):
            p.weight = weights[i]
        self.getParticleVariance()

    def getParticleVariance(self):
        variance_x = np.var([p.x for p in self.particles])
        variance_y = np.var([p.y for p in self.particles])
        self.variance = variance_x + variance_y
        return self.variance

    def resampleParticle(self):
        particles_new = []
        weights = [p.weight for p in self.particles]
        self.sorted_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)
        cumulative_sum = np.cumsum(weights)
        self.getParticleVariance()
        print("Current Variance: ", self.variance)

        for i in range(self.num_particles):
            rand_val = random.uniform(0, cumulative_sum[-1])
            idx = np.searchsorted(cumulative_sum, rand_val)
            chosen_particle = self.particles[idx]

            if self.variance > 150:
                heading_noise = np.random.normal(0, 2*np.pi*0.06)
                y_noise = np.random.normal(0, 0.25)
                x_noise = np.random.normal(0, 0.25)
            else:
                heading_noise = np.random.normal(0, 0.1)
                x_noise = np.random.normal(0, 0.2)
                y_noise = np.random.normal(0, 0.2)

            new_particle = Particle(
                x=chosen_particle.x + x_noise,
                y=chosen_particle.y + y_noise,
                heading=(chosen_particle.heading + heading_noise) % (2 * np.pi),
                maze=self.world,
                sensor_limit=self.sensor_limit,
                noisy=False
            )
            particles_new.append(new_particle)

        self.particles = particles_new

    def particleMotionModel(self):
        for p in self.particles:
            p.x, p.y, p.heading = self.predictMotion(p.x, p.y, p.heading)
        self.control = []

    def predictMotion(self, x, y, heading):
        dt = 0.01
        for (vr, delta) in self.control:
            dx, dy, dheading = vehicle_dynamics(dt, [x, y, heading], vr, delta)
            x += dx
            y += dy
            heading += dheading
        return (x, y, heading)

    def runFilter(self, save_dir='./results', duration=60):
        """
        Run particle filter for specified duration and save results
        Args:
            save_dir: Directory to save plots (default: ./results)
            duration: Run duration in seconds (default: 60)
        """
        # Create directory if doesn't exist
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize tracking variables
        count = 0
        timestep = []
        error_pos = []
        error_head = []
        start_time = time.time()
        
        # Set up plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        pos_err_line, = ax1.plot([], [], 'b', label='Position Error')
        head_err_line, = ax2.plot([], [], 'r', label='Heading Error')
        
        # Configure plots
        ax1.set_title("Position Error (meters) vs time")
        ax2.set_title("Heading Error (radians) vs time")
        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.show(block=False)
        
        # Main loop
        while time.time() - start_time < duration:
            try:
                # Particle filter steps
                count += 1
                self.particleMotionModel()
                reading = self.bob.read_sensor()
                self.updateWeight(reading)
                self.resampleParticle()
                
                # Visualization
                self.world.clear_objects()
                self.world.show_particles(self.particles)
                self.world.show_robot(self.bob)
                self.world.show_estimated_location(self.particles)
                
                # Error calculation
                x_error = self.bob.x - self.sorted_particles[0].x
                y_error = self.bob.y - self.sorted_particles[0].y
                heading_error = self.bob.heading - self.sorted_particles[0].heading
                error_pos.append(np.linalg.norm([x_error, y_error]))
                error_head.append(heading_error)
                timestep.append(count)
                
                # Update plots
                pos_err_line.set_data(timestep, error_pos)
                head_err_line.set_data(timestep, np.unwrap(error_head))
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                
                # Refresh display
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
                
            except KeyboardInterrupt:
                break
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fig.savefig(f"{save_dir}/particle_filter_results_{timestamp}.png")
        plt.close()