"""
Full robot geometry tracking for collision avoidance.

This module provides collision checking between the entire robot arm
and obstacles, not just the end-effector.
"""
import dataclasses
import logging
from typing import Any, Dict, Optional, Tuple, List
import numpy as np


@dataclasses.dataclass
class RobotLinkGeometry:
    """Approximated geometry for a robot link."""
    name: str
    geom_id: int
    original_type: int  # 7 = MESH, 6 = BOX, etc.
    
    # Bounding capsule approximation (for meshes)
    # Capsule axis is along local Z by convention
    capsule_radius: float
    capsule_half_length: float
    
    # Or box (for actual boxes like finger pads)
    box_half_extents: Optional[np.ndarray] = None


@dataclasses.dataclass  
class RobotObstacleDistance:
    """Distance result between robot and obstacle."""
    robot_geom_name: str
    obstacle_name: str
    distance: float
    robot_surface_point: np.ndarray
    obstacle_surface_point: np.ndarray


class FullRobotGeometryTracker:
    """
    Tracks full robot arm geometry using bounding primitive approximations.
    
    Meshes are approximated as capsules aligned with the link's local Z axis.
    """
    
    # MuJoCo geometry types
    PLANE = 0
    HFIELD = 1
    SPHERE = 2
    CAPSULE = 3
    ELLIPSOID = 4
    CYLINDER = 5
    BOX = 6
    MESH = 7
    
    # Robot collision geoms to track (skip link0 - it's the base)
    ROBOT_COLLISION_GEOMS = [
        # "robot0_link1_collision",
        # "robot0_link2_collision", 
        # "robot0_link3_collision",
        # "robot0_link4_collision",
        # "robot0_link5_collision",
        # "robot0_link6_collision",
        # "robot0_link7_collision",
        "gripper0_hand_collision",
        "gripper0_finger1_collision",
        "gripper0_finger2_collision",
        "gripper0_finger1_pad_collision",
        "gripper0_finger2_pad_collision",
    ]
    
    # Manual bounding capsule parameters for Franka Panda (approximate)
    # Format: (radius, half_length) - tune these based on actual robot
    LINK_CAPSULE_PARAMS = {
        # "robot0_link1_collision": (0.06, 0.10),
        # "robot0_link2_collision": (0.06, 0.10),
        # "robot0_link3_collision": (0.05, 0.12),
        # "robot0_link4_collision": (0.05, 0.10),
        # "robot0_link5_collision": (0.05, 0.12),
        # "robot0_link6_collision": (0.05, 0.08),
        # "robot0_link7_collision": (0.05, 0.06),
        "gripper0_hand_collision": (0.04, 0.06),
        "gripper0_finger1_collision": (0.02, 0.04),
        "gripper0_finger2_collision": (0.02, 0.04),
    }
    
    def __init__(self, env, verbose: bool = False, safety_margin: float = 0.04):
        self.env = env
        self.verbose = verbose
        self.safety_margin = safety_margin
        
        self.robot_geoms: Dict[str, RobotLinkGeometry] = {}
        self.obstacle_geoms: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_robot_geoms()
        self._initialize_obstacle_geoms()
        
        if verbose:
            self._print_geometry_info()
    
    def _initialize_robot_geoms(self):
        """Initialize robot collision geometries with bounding approximations."""
        sim = self.env.sim
        model = sim.model
        
        for geom_name in self.ROBOT_COLLISION_GEOMS:
            try:
                geom_id = model.geom_name2id(geom_name)
            except KeyError:
                if self.verbose:
                    logging.warning(f"Robot geom not found: {geom_name}")
                continue
            
            geom_type = model.geom_type[geom_id]
            geom_size = model.geom_size[geom_id].copy()
            
            if geom_type == self.BOX:
                # Actual box (finger pads)
                self.robot_geoms[geom_name] = RobotLinkGeometry(
                    name=geom_name,
                    geom_id=geom_id,
                    original_type=geom_type,
                    capsule_radius=0.0,
                    capsule_half_length=0.0,
                    box_half_extents=geom_size[:3].copy(),
                )
            elif geom_type == self.MESH:
                # Approximate mesh with capsule
                if geom_name in self.LINK_CAPSULE_PARAMS:
                    radius, half_len = self.LINK_CAPSULE_PARAMS[geom_name]
                else:
                    # Fallback: use reasonable defaults
                    radius, half_len = 0.05, 0.08
                    if self.verbose:
                        logging.warning(f"No capsule params for {geom_name}, using defaults")
                
                self.robot_geoms[geom_name] = RobotLinkGeometry(
                    name=geom_name,
                    geom_id=geom_id,
                    original_type=geom_type,
                    capsule_radius=radius,
                    capsule_half_length=half_len,
                    box_half_extents=None,
                )
            else:
                # Other primitive types - approximate as capsule
                if geom_name in self.LINK_CAPSULE_PARAMS:
                    radius, half_len = self.LINK_CAPSULE_PARAMS[geom_name]
                else:
                    radius, half_len = 0.05, 0.08
                
                self.robot_geoms[geom_name] = RobotLinkGeometry(
                    name=geom_name,
                    geom_id=geom_id,
                    original_type=geom_type,
                    capsule_radius=radius,
                    capsule_half_length=half_len,
                    box_half_extents=None,
                )
    
    def _initialize_obstacle_geoms(self):
        """Find obstacle geometries in the scene."""
        sim = self.env.sim
        model = sim.model
        
        for geom_id in range(model.ngeom):
            geom_name = model.geom_id2name(geom_id)
            if geom_name and "obstacle" in geom_name.lower():
                self.obstacle_geoms[geom_name] = {
                    'geom_id': geom_id,
                    'geom_type': model.geom_type[geom_id],
                    'size': model.geom_size[geom_id].copy(),
                }
    
    def _print_geometry_info(self):
        """Print diagnostic info about tracked geometries."""
        print("\n=== Robot Collision Geometries ===")
        for name, geom in self.robot_geoms.items():
            if geom.box_half_extents is not None:
                print(f"  {name}: BOX half_extents={geom.box_half_extents}")
            else:
                print(f"  {name}: CAPSULE r={geom.capsule_radius:.3f} h={geom.capsule_half_length:.3f}")
        
        print("\n=== Obstacle Geometries ===")
        type_names = {
            self.SPHERE: 'SPHERE', 
            self.CAPSULE: 'CAPSULE', 
            self.CYLINDER: 'CYLINDER', 
            self.BOX: 'BOX', 
            self.MESH: 'MESH'
        }
        for name, data in self.obstacle_geoms.items():
            t = type_names.get(data['geom_type'], str(data['geom_type']))
            print(f"  {name}: {t} size={data['size']}")
        print()
    
    def get_geom_pose(self, geom_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get world position and rotation matrix of a geom."""
        sim = self.env.sim
        pos = sim.data.geom_xpos[geom_id].copy()
        rot = sim.data.geom_xmat[geom_id].reshape(3, 3).copy()
        return pos, rot
    
    # =========================================================================
    # Distance computation functions
    # =========================================================================
    
    def closest_point_capsule_to_point(
        self,
        cap_pos: np.ndarray,
        cap_rot: np.ndarray, 
        cap_radius: float,
        cap_half_len: float,
        point: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Distance from capsule to point.
        Returns (distance, closest_point_on_capsule_surface).
        """
        # Capsule axis in world frame (MuJoCo capsules are along local Z)
        axis = cap_rot @ np.array([0.0, 0.0, 1.0])
        p1 = cap_pos - axis * cap_half_len  # Bottom endpoint
        p2 = cap_pos + axis * cap_half_len  # Top endpoint
        
        # Project point onto capsule axis
        v = p2 - p1
        w = point - p1
        
        t = np.dot(w, v) / (np.dot(v, v) + 1e-8)
        t = np.clip(t, 0.0, 1.0)
        
        # Closest point on axis
        closest_on_axis = p1 + t * v
        
        # Direction from axis to point
        to_point = point - closest_on_axis
        dist_to_axis = np.linalg.norm(to_point)
        
        if dist_to_axis < 1e-8:
            # Point is on the axis
            direction = cap_rot @ np.array([1.0, 0.0, 0.0])  # Arbitrary perpendicular
        else:
            direction = to_point / dist_to_axis
        
        # Surface point
        surface_point = closest_on_axis + direction * cap_radius
        
        # Distance (negative if inside capsule)
        distance = dist_to_axis - cap_radius
        
        return distance, surface_point
    
    def closest_point_box_to_point(
        self,
        box_pos: np.ndarray,
        box_rot: np.ndarray,
        box_half_extents: np.ndarray,
        point: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Distance from box to point.
        Returns (distance, closest_point_on_box_surface).
        """
        # Transform point to box local frame
        local_point = box_rot.T @ (point - box_pos)
        
        # Clamp to box
        clamped = np.clip(local_point, -box_half_extents, box_half_extents)
        
        # Check if inside
        inside = np.all(np.abs(local_point) <= box_half_extents)
        
        if inside:
            # Find closest face
            distances_to_faces = box_half_extents - np.abs(local_point)
            closest_axis = np.argmin(distances_to_faces)
            clamped[closest_axis] = np.sign(local_point[closest_axis]) * box_half_extents[closest_axis]
            surface_point_local = clamped
            distance = -distances_to_faces[closest_axis]
        else:
            surface_point_local = clamped
            distance = np.linalg.norm(local_point - clamped)
        
        # Transform back to world
        surface_point = box_rot @ surface_point_local + box_pos
        
        return distance, surface_point
    
    def capsule_to_box_distance(
        self,
        cap_pos: np.ndarray, cap_rot: np.ndarray, 
        cap_radius: float, cap_half_len: float,
        box_pos: np.ndarray, box_rot: np.ndarray,
        box_half_extents: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Distance between capsule and box.
        Returns (distance, capsule_surface_point, box_surface_point).
        """
        # Capsule axis endpoints
        axis = cap_rot @ np.array([0.0, 0.0, 1.0])
        p1 = cap_pos - axis * cap_half_len
        p2 = cap_pos + axis * cap_half_len
        
        # Sample points along capsule axis and find minimum distance to box
        best_dist = float('inf')
        best_cap_pt = cap_pos.copy()
        best_box_pt = box_pos.copy()
        
        n_samples = 5
        for i in range(n_samples):
            t = i / (n_samples - 1) if n_samples > 1 else 0.5
            axis_pt = p1 + t * (p2 - p1)
            
            # Distance from this axis point to box
            dist_to_box, box_surface = self.closest_point_box_to_point(
                box_pos, box_rot, box_half_extents, axis_pt
            )
            
            # Account for capsule radius
            total_dist = dist_to_box - cap_radius
            
            if total_dist < best_dist:
                best_dist = total_dist
                best_box_pt = box_surface
                
                # Capsule surface point
                direction = axis_pt - box_surface
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-8:
                    direction = direction / dir_norm
                else:
                    direction = np.array([1.0, 0.0, 0.0])
                best_cap_pt = axis_pt - direction * cap_radius
        
        return best_dist, best_cap_pt, best_box_pt
    
    def capsule_to_sphere_distance(
        self,
        cap_pos: np.ndarray, cap_rot: np.ndarray,
        cap_radius: float, cap_half_len: float,
        sphere_pos: np.ndarray, sphere_radius: float,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Distance between capsule and sphere."""
        dist_to_center, cap_surface = self.closest_point_capsule_to_point(
            cap_pos, cap_rot, cap_radius, cap_half_len, sphere_pos
        )
        
        # The returned dist_to_center is from capsule surface to sphere center
        # We need to subtract sphere radius
        distance = dist_to_center - sphere_radius
        
        # Sphere surface point
        direction = sphere_pos - cap_surface
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-8:
            sphere_surface = sphere_pos - (direction / dir_norm) * sphere_radius
        else:
            sphere_surface = sphere_pos + np.array([sphere_radius, 0.0, 0.0])
        
        return distance, cap_surface, sphere_surface
    
    def capsule_to_cylinder_distance(
        self,
        cap_pos: np.ndarray, cap_rot: np.ndarray,
        cap_radius: float, cap_half_len: float,
        cyl_pos: np.ndarray, cyl_rot: np.ndarray,
        cyl_radius: float, cyl_half_height: float,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Distance between capsule and cylinder.
        Uses line segment closest points algorithm.
        """
        # Capsule axis endpoints
        cap_axis = cap_rot @ np.array([0.0, 0.0, 1.0])
        cap_p1 = cap_pos - cap_axis * cap_half_len
        cap_p2 = cap_pos + cap_axis * cap_half_len
        
        # Cylinder axis endpoints
        cyl_axis = cyl_rot @ np.array([0.0, 0.0, 1.0])
        cyl_p1 = cyl_pos - cyl_axis * cyl_half_height
        cyl_p2 = cyl_pos + cyl_axis * cyl_half_height
        
        # Find closest points between the two line segments
        cap_pt, cyl_pt = self._closest_points_segments(cap_p1, cap_p2, cyl_p1, cyl_p2)
        
        # Distance between axis points
        axis_dist = np.linalg.norm(cap_pt - cyl_pt)
        
        # Surface distance
        distance = axis_dist - cap_radius - cyl_radius
        
        # Surface points
        direction = cyl_pt - cap_pt
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-8:
            direction = direction / dir_norm
        else:
            direction = np.array([1.0, 0.0, 0.0])
        
        cap_surface = cap_pt + direction * cap_radius
        cyl_surface = cyl_pt - direction * cyl_radius
        
        return distance, cap_surface, cyl_surface
    
    def box_to_box_distance(
        self,
        box1_pos: np.ndarray, box1_rot: np.ndarray, box1_half: np.ndarray,
        box2_pos: np.ndarray, box2_rot: np.ndarray, box2_half: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Distance between two boxes.
        Uses sampling approximation for simplicity.
        """
        best_dist = float('inf')
        best_pt1 = box1_pos.copy()
        best_pt2 = box2_pos.copy()
        
        # Generate sample points on box1 (corners and face centers)
        samples = []
        for sx in [-1, 0, 1]:
            for sy in [-1, 0, 1]:
                for sz in [-1, 0, 1]:
                    if sx == 0 and sy == 0 and sz == 0:
                        continue
                    local_pt = np.array([
                        sx * box1_half[0], 
                        sy * box1_half[1], 
                        sz * box1_half[2]
                    ])
                    world_pt = box1_rot @ local_pt + box1_pos
                    samples.append(world_pt)
        
        for pt1 in samples:
            dist, pt2 = self.closest_point_box_to_point(box2_pos, box2_rot, box2_half, pt1)
            if dist < best_dist:
                best_dist = dist
                best_pt1 = pt1
                best_pt2 = pt2
        
        return best_dist, best_pt1, best_pt2
    
    def _closest_points_segments(
        self,
        p1: np.ndarray, q1: np.ndarray,
        p2: np.ndarray, q2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find closest points between two 3D line segments.
        Segment 1: p1 to q1
        Segment 2: p2 to q2
        Returns (point_on_seg1, point_on_seg2).
        """
        d1 = q1 - p1  # Direction of segment 1
        d2 = q2 - p2  # Direction of segment 2
        r = p1 - p2
        
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        # Check if both segments are points
        if a < 1e-8 and e < 1e-8:
            return p1.copy(), p2.copy()
        
        if a < 1e-8:
            # Segment 1 is a point
            s = 0.0
            t = np.clip(f / e, 0.0, 1.0)
        else:
            c = np.dot(d1, r)
            if e < 1e-8:
                # Segment 2 is a point
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            else:
                b = np.dot(d1, d2)
                denom = a * e - b * b
                
                if abs(denom) > 1e-8:
                    s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
                else:
                    s = 0.0
                
                t = (b * s + f) / e
                
                if t < 0:
                    t = 0.0
                    s = np.clip(-c / a, 0.0, 1.0)
                elif t > 1:
                    t = 1.0
                    s = np.clip((b - c) / a, 0.0, 1.0)
        
        closest1 = p1 + s * d1
        closest2 = p2 + t * d2
        
        return closest1, closest2
    
    # =========================================================================
    # Main interface
    # =========================================================================
    
    def get_min_robot_obstacle_distance(self) -> Optional[RobotObstacleDistance]:
        """
        Find minimum distance between any robot geometry and any obstacle.
        Returns detailed info about the closest pair.
        """
        if not self.robot_geoms or not self.obstacle_geoms:
            return None
        
        min_result = None
        min_distance = float('inf')
        
        for robot_name, robot_geom in self.robot_geoms.items():
            robot_pos, robot_rot = self.get_geom_pose(robot_geom.geom_id)
            
            for obs_name, obs_data in self.obstacle_geoms.items():
                # obs_pos, obs_rot = self.get_geom_pose(obs_data['geom_id'])
                obs_offset = obs_data.get('pos_offset', None)
                obs_pos, obs_rot = self.get_geom_pose(obs_data['geom_id'], obs_offset)
                obs_type = obs_data['geom_type']
                obs_size = obs_data['size']
                
                # Compute distance based on robot geometry type
                if robot_geom.box_half_extents is not None:
                    # Robot part is a box (finger pads)
                    distance, robot_pt, obs_pt = self._robot_box_to_obstacle(
                        robot_pos, robot_rot, robot_geom.box_half_extents,
                        obs_type, obs_pos, obs_rot, obs_size
                    )
                else:
                    # Robot part is approximated as capsule
                    distance, robot_pt, obs_pt = self._robot_capsule_to_obstacle(
                        robot_pos, robot_rot, 
                        robot_geom.capsule_radius, robot_geom.capsule_half_length,
                        obs_type, obs_pos, obs_rot, obs_size
                    )
                
                if distance < min_distance:
                    min_distance = distance
                    min_result = RobotObstacleDistance(
                        robot_geom_name=robot_name,
                        obstacle_name=obs_name,
                        distance=distance,
                        robot_surface_point=robot_pt,
                        obstacle_surface_point=obs_pt,
                    )
        
        return min_result
    
    def get_all_robot_obstacle_distances(self) -> List[RobotObstacleDistance]:
        """
        Get distances for all robot-obstacle pairs.
        Useful for visualization or detailed analysis.
        """
        results = []
        
        if not self.robot_geoms or not self.obstacle_geoms:
            return results
        
        for robot_name, robot_geom in self.robot_geoms.items():
            robot_pos, robot_rot = self.get_geom_pose(robot_geom.geom_id)
            
            for obs_name, obs_data in self.obstacle_geoms.items():
                obs_pos, obs_rot = self.get_geom_pose(obs_data['geom_id'])
                obs_type = obs_data['geom_type']
                obs_size = obs_data['size']
                
                if robot_geom.box_half_extents is not None:
                    distance, robot_pt, obs_pt = self._robot_box_to_obstacle(
                        robot_pos, robot_rot, robot_geom.box_half_extents,
                        obs_type, obs_pos, obs_rot, obs_size
                    )
                else:
                    distance, robot_pt, obs_pt = self._robot_capsule_to_obstacle(
                        robot_pos, robot_rot, 
                        robot_geom.capsule_radius, robot_geom.capsule_half_length,
                        obs_type, obs_pos, obs_rot, obs_size
                    )
                
                results.append(RobotObstacleDistance(
                    robot_geom_name=robot_name,
                    obstacle_name=obs_name,
                    distance=distance,
                    robot_surface_point=robot_pt,
                    obstacle_surface_point=obs_pt,
                ))
        
        return results
    
    def _robot_capsule_to_obstacle(
        self,
        cap_pos: np.ndarray, cap_rot: np.ndarray,
        cap_radius: float, cap_half_len: float,
        obs_type: int, obs_pos: np.ndarray, obs_rot: np.ndarray, obs_size: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Dispatch capsule-to-obstacle distance based on obstacle type."""
        
        if obs_type == self.SPHERE:
            return self.capsule_to_sphere_distance(
                cap_pos, cap_rot, cap_radius, cap_half_len,
                obs_pos, obs_size[0]
            )
        elif obs_type == self.BOX:
            return self.capsule_to_box_distance(
                cap_pos, cap_rot, cap_radius, cap_half_len,
                obs_pos, obs_rot, obs_size[:3]
            )
        elif obs_type == self.CYLINDER:
            return self.capsule_to_cylinder_distance(
                cap_pos, cap_rot, cap_radius, cap_half_len,
                obs_pos, obs_rot, obs_size[0], obs_size[1]
            )
        elif obs_type == self.MESH:
            # Approximate mesh obstacle as box using its size
            # Note: This is a rough approximation
            return self.capsule_to_box_distance(
                cap_pos, cap_rot, cap_radius, cap_half_len,
                obs_pos, obs_rot, obs_size[:3]
            )
        elif obs_type == self.CAPSULE:
            # Treat obstacle capsule similar to cylinder
            return self.capsule_to_cylinder_distance(
                cap_pos, cap_rot, cap_radius, cap_half_len,
                obs_pos, obs_rot, obs_size[0], obs_size[1]
            )
        else:
            # Fallback: point-to-capsule distance
            dist, surface = self.closest_point_capsule_to_point(
                cap_pos, cap_rot, cap_radius, cap_half_len, obs_pos
            )
            return dist, surface, obs_pos.copy()
    
    def _robot_box_to_obstacle(
        self,
        box_pos: np.ndarray, box_rot: np.ndarray, box_half: np.ndarray,
        obs_type: int, obs_pos: np.ndarray, obs_rot: np.ndarray, obs_size: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Dispatch box-to-obstacle distance."""
        
        if obs_type == self.BOX or obs_type == self.MESH:
            return self.box_to_box_distance(
                box_pos, box_rot, box_half,
                obs_pos, obs_rot, obs_size[:3]
            )
        elif obs_type == self.SPHERE:
            dist, surface = self.closest_point_box_to_point(
                box_pos, box_rot, box_half, obs_pos
            )
            # Adjust for sphere radius
            direction = obs_pos - surface
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-8:
                obs_surface = obs_pos - (direction / dir_norm) * obs_size[0]
            else:
                obs_surface = obs_pos.copy()
            return dist - obs_size[0], surface, obs_surface
        elif obs_type == self.CYLINDER:
            # Approximate: sample box corners against cylinder
            # For better accuracy, would need proper box-cylinder distance
            best_dist = float('inf')
            best_box_pt = box_pos.copy()
            best_cyl_pt = obs_pos.copy()
            
            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    for sz in [-1, 1]:
                        local_pt = np.array([sx * box_half[0], sy * box_half[1], sz * box_half[2]])
                        world_pt = box_rot @ local_pt + box_pos
                        
                        # Distance from point to cylinder
                        dist, cyl_pt = self._point_to_cylinder_distance(
                            world_pt, obs_pos, obs_rot, obs_size[0], obs_size[1]
                        )
                        
                        if dist < best_dist:
                            best_dist = dist
                            best_box_pt = world_pt
                            best_cyl_pt = cyl_pt
            
            return best_dist, best_box_pt, best_cyl_pt
        else:
            # Fallback: point distance
            dist, surface = self.closest_point_box_to_point(
                box_pos, box_rot, box_half, obs_pos
            )
            return dist, surface, obs_pos.copy()
    
    def _point_to_cylinder_distance(
        self,
        point: np.ndarray,
        cyl_pos: np.ndarray, cyl_rot: np.ndarray,
        cyl_radius: float, cyl_half_height: float,
    ) -> Tuple[float, np.ndarray]:
        """Distance from point to cylinder surface."""
        # Transform to cylinder local frame
        local_point = cyl_rot.T @ (point - cyl_pos)
        
        radial = local_point[:2]
        axial = local_point[2]
        radial_dist = np.linalg.norm(radial)
        
        # Clamp axial coordinate
        clamped_axial = np.clip(axial, -cyl_half_height, cyl_half_height)
        
        # Check if inside or outside
        inside_radial = radial_dist <= cyl_radius
        inside_axial = np.abs(axial) <= cyl_half_height
        
        if inside_radial and inside_axial:
            # Inside cylinder - find closest surface
            dist_to_curved = cyl_radius - radial_dist
            dist_to_cap = cyl_half_height - np.abs(axial)
            
            if dist_to_curved < dist_to_cap:
                # Closest to curved surface
                if radial_dist > 1e-8:
                    clamped_radial = radial / radial_dist * cyl_radius
                else:
                    clamped_radial = np.array([cyl_radius, 0.0])
                local_surface = np.array([clamped_radial[0], clamped_radial[1], axial])
                distance = -dist_to_curved
            else:
                # Closest to cap
                local_surface = np.array([radial[0], radial[1], np.sign(axial) * cyl_half_height])
                distance = -dist_to_cap
        else:
            # Outside - find closest point
            if radial_dist > 1e-8:
                clamped_radial = radial / radial_dist * min(radial_dist, cyl_radius)
            else:
                clamped_radial = np.array([0.0, 0.0])
            
            local_surface = np.array([clamped_radial[0], clamped_radial[1], clamped_axial])
            world_surface = cyl_rot @ local_surface + cyl_pos
            distance = np.linalg.norm(point - world_surface)
            return distance, world_surface
        
        world_surface = cyl_rot @ local_surface + cyl_pos
        return distance, world_surface

# Add to FullRobotGeometryTracker:
# Add to FullRobotGeometryTracker:

    def add_virtual_obstacle(self, name: str, reference_geom: str, size: np.ndarray, pos_offset: np.ndarray = None):
        """Add a virtual obstacle with custom size, optionally offset from a reference geom."""
        model = self.env.sim.model
        ref_id = model.geom_name2id(reference_geom)
        
        self.obstacle_geoms[name] = {
            'geom_id': ref_id,
            'geom_type': self.BOX,
            'size': size.copy(),
            'pos_offset': pos_offset if pos_offset is not None else np.zeros(3),
            'virtual': True,
        }

    def get_geom_pose(self, geom_id: int, pos_offset: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get world position and rotation matrix of a geom."""
        sim = self.env.sim
        pos = sim.data.geom_xpos[geom_id].copy()
        rot = sim.data.geom_xmat[geom_id].reshape(3, 3).copy()
        if pos_offset is not None:
            pos += rot @ pos_offset  # offset in local frame
        return pos, rot
    def add_obstacle_geom(self, geom_name: str):
        """Register an additional geom as an obstacle."""
        model = self.env.sim.model
        try:
            geom_id = model.geom_name2id(geom_name)
        except KeyError:
            logging.warning(f"Geom not found: {geom_name}")
            return
        
        self.obstacle_geoms[geom_name] = {
            'geom_id': geom_id,
            'geom_type': model.geom_type[geom_id],
            'size': model.geom_size[geom_id].copy(),
        }
        if self.verbose:
            logging.info(f"Added obstacle geom: {geom_name}")
# =============================================================================
# Compatibility wrapper for existing code
# =============================================================================

@dataclasses.dataclass 
class GroundTruthObstacleInfo:
    """
    Ground truth obstacle information.
    Compatibility class for existing visualization code.
    """
    name: str
    geom_type: int
    position: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    surface_point: np.ndarray
    distance: float


def robot_obstacle_to_ground_truth_info(
    robot_obs_dist: RobotObstacleDistance,
) -> GroundTruthObstacleInfo:
    """Convert RobotObstacleDistance to GroundTruthObstacleInfo for compatibility."""
    return GroundTruthObstacleInfo(
        name=robot_obs_dist.obstacle_name,
        geom_type=0,  # Not used by visualization
        position=robot_obs_dist.obstacle_surface_point,
        size=np.zeros(3),
        rotation=np.eye(3),
        surface_point=robot_obs_dist.obstacle_surface_point,
        distance=robot_obs_dist.distance,
    )