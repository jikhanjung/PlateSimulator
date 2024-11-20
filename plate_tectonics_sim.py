import numpy as np
from scipy.ndimage import gaussian_filter

class PlateTectonicsSimulator:
    def __init__(self, grid_size=100, num_plates=5):
        self.grid_size = grid_size
        self.num_plates = num_plates
        
        self.movement_scale = 0.0002
        self.thermal_diffusivity = 1e-8
        self.mantle_temp = 1400.0
        self.surface_temp = 300.0
        
        self.elevation = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.temperature = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.plate_ids = np.zeros((grid_size, grid_size), dtype=int)
        self.plate_thickness = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        self._initialize_plates()
        self._initialize_temperature()
    
    def _initialize_plates(self):
        # Create plate centers with minimum separation
        centers = []
        while len(centers) < self.num_plates:
            point = np.random.rand(2) * self.grid_size
            if not centers or all(np.linalg.norm(point - c) > self.grid_size/3 for c in centers):
                centers.append(point)
        centers = np.array(centers)
        
        # Assign initial plate configurations
        distances = np.zeros((self.num_plates, self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for p in range(self.num_plates):
                    distances[p,i,j] = np.linalg.norm(centers[p] - [i,j])
        
        self.plate_ids = np.argmin(distances, axis=0)
        
        # Initialize plate properties
        for plate_id in range(self.num_plates):
            mask = self.plate_ids == plate_id
            self.plate_thickness[mask] = np.random.uniform(30, 70)
            self.elevation[mask] = np.random.uniform(-2, 2)
        
        # Initialize plate velocities with convergent/divergent zones
        angles = np.linspace(0, 2*np.pi, self.num_plates, endpoint=False)
        angles += np.random.uniform(-0.1, 0.1, self.num_plates)
        
        # Create convergent boundaries by making adjacent plates move toward each other
        speeds = np.zeros(self.num_plates)
        for i in range(self.num_plates):
            if i % 2 == 0:
                speeds[i] = np.random.uniform(0.02, 0.08)
            else:
                speeds[i] = -np.random.uniform(0.02, 0.08)
        
        self.plate_velocities = np.column_stack((
            speeds * np.cos(angles),
            speeds * np.sin(angles)
        )) * self.movement_scale
    
    def _initialize_temperature(self):
        # Temperature varies with depth (elevation)
        depth_normalized = (self.elevation - np.min(self.elevation)) / (np.max(self.elevation) - np.min(self.elevation))
        self.temperature = self.surface_temp + (self.mantle_temp - self.surface_temp) * (1 - depth_normalized)
    
    def _calculate_convergence(self, plate1, plate2):
        vel1 = self.plate_velocities[plate1]
        vel2 = self.plate_velocities[plate2]
        relative_vel = vel1 - vel2
        return np.dot(relative_vel, relative_vel)
    
    def simulate_collision(self):
        gradx = np.gradient(self.plate_ids, axis=0)
        grady = np.gradient(self.plate_ids, axis=1)
        boundary_mask = (np.abs(gradx) + np.abs(grady)) > 0
        
        deformation = np.zeros_like(self.elevation)
        heating = np.zeros_like(self.temperature)
        thickness_change = np.zeros_like(self.plate_thickness)
        
        for i in range(1, self.grid_size-1):
            for j in range(1, self.grid_size-1):
                if boundary_mask[i,j]:
                    plate1 = self.plate_ids[i,j] % self.num_plates
                    plate2 = self.plate_ids[min(i+1, self.grid_size-1),j] % self.num_plates
                    
                    convergence = self._calculate_convergence(plate1, plate2)
                    
                    if convergence > 0:  # Convergent boundary
                        # Determine which plate subducts based on thickness
                        if self.plate_thickness[i,j] < self.plate_thickness[i+1,j]:
                            # Current plate subducts
                            deformation[i,j] = -convergence * 2.0
                            thickness_change[i,j] = -convergence * 0.5
                            heating[i,j] = convergence * 200
                        else:
                            # Next plate subducts
                            deformation[i,j] = convergence * 2.0
                            thickness_change[i,j] = convergence * 0.5
                            heating[i+1,j] = convergence * 200
                    else:  # Divergent boundary
                        deformation[i,j] = -abs(convergence) * 0.5
                        thickness_change[i,j] = abs(convergence) * 0.2
                        heating[i,j] = abs(convergence) * 50
        
        self.elevation += gaussian_filter(deformation, sigma=1)
        self.temperature += gaussian_filter(heating, sigma=1)
        self.plate_thickness += gaussian_filter(thickness_change, sigma=1)
        
        # Ensure temperature stays within realistic bounds
        self.temperature = np.clip(self.temperature, self.surface_temp, self.mantle_temp)
    
    def step(self, dt=50):
        new_elevation = np.zeros_like(self.elevation)
        new_temperature = np.zeros_like(self.temperature)
        new_plate_ids = np.zeros_like(self.plate_ids)
        new_thickness = np.zeros_like(self.plate_thickness)
        
        for plate_id in range(self.num_plates):
            mask = (self.plate_ids == plate_id)
            vel = self.plate_velocities[plate_id] * dt
            
            y, x = np.mgrid[0:self.grid_size, 0:self.grid_size]
            coords = np.column_stack((y.ravel(), x.ravel()))
            shift = np.array([vel[0], vel[1]])
            new_coords = coords - shift
            new_coords = new_coords % self.grid_size
            
            y_indices = new_coords[:,0].reshape(self.grid_size, self.grid_size)
            x_indices = new_coords[:,1].reshape(self.grid_size, self.grid_size)
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if mask[i,j]:
                        new_i = int(y_indices[i,j])
                        new_j = int(x_indices[i,j])
                        new_plate_ids[new_i, new_j] = plate_id
                        new_elevation[new_i, new_j] = self.elevation[i,j]
                        new_temperature[new_i, new_j] = self.temperature[i,j]
                        new_thickness[new_i, new_j] = self.plate_thickness[i,j]
        
        self.plate_ids = new_plate_ids
        self.elevation = new_elevation
        self.temperature = new_temperature
        self.plate_thickness = new_thickness
        
        self.simulate_collision()
        
        # Thermal diffusion
        temp_update = self.thermal_diffusivity * dt * (
            np.roll(self.temperature, 1, axis=0) + 
            np.roll(self.temperature, -1, axis=0) + 
            np.roll(self.temperature, 1, axis=1) + 
            np.roll(self.temperature, -1, axis=1) - 
            4 * self.temperature
        )
        self.temperature += temp_update