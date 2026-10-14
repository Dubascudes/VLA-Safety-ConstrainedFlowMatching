from __future__ import annotations

# from openpi.examples.aloha_sim import env

"""
SafeLIBERO evaluation with Safety Visualization and LIVE DISPLAY

This version includes:
- Obstacle zone visualization in saved videos
- Real-time live display of robot trials using OpenCV
- **Dropdown menus for task suite and task ID selection**
- **Interactive sliders for safety margin and guidance strength**

Usage:
    Terminal 1 (Server):
        uv run scripts/serve_policy.py --env LIBERO
    
    Terminal 2 (Evaluation):
        # With live display (requires X11/display):
        python main_with_live_display.py --live-display
        
        # With live display at specific FPS:
        python main_with_live_display.py --live-display --display-fps 30
        
        # Save videos AND show live display:
        python main_with_live_display.py --live-display --save-videos
        
        # Headless (no display, just save videos):
        python main_with_live_display.py --save-videos

Controls during live display:
    - 'q' or ESC: Quit current episode
    - SPACE: Pause/resume
    - 'n': Skip to next episode
    - 's': Screenshot current frame
    - Click dropdown menus to change task suite or task ID
    - Drag sliders to adjust safety margin and guidance strength
"""

import collections
# import dataclasses
from dataclasses import dataclass
import logging
import math
import pathlib
import time
from typing import Literal, Optional, Dict, Any, List, Tuple, Callable

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from proprioception import FullRobotGeometryTracker

# Try to import visualization (optional)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logging.warning("cv2 not available, visualization will be limited")


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 1024

MAX_STEPS = {
    "safelibero_spatial": 300,
    "safelibero_object": 600,
    "safelibero_goal": 600,
    "safelibero_long": 1200,
}

# Available task suites
TASK_SUITES = [
    "safelibero_spatial",
    "safelibero_object", 
    "safelibero_goal",
    "safelibero_long",
]


@dataclass
class ObstacleGeometry:
    """Obstacle described by its primitive geometry."""
    geom_type: int
    position: np.ndarray
    rotation: np.ndarray
    size: np.ndarray

# =============================================================================
# Slider Component
# =============================================================================

class Slider:
    """A horizontal slider component for OpenCV UI."""
    
    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        initial_val: float,
        x: int,
        y: int,
        width: int = 150,
        height: int = 20,
        format_str: str = "{:.3f}",
        on_change: Optional[Callable[[float], None]] = None,
    ):
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self._value = initial_val
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.format_str = format_str
        self.on_change = on_change
        
        self.is_dragging = False
        self.hover = False
        
        # Colors
        self.track_color = (80, 80, 80)
        self.fill_color = (100, 160, 220)
        self.fill_hover_color = (120, 180, 240)
        self.knob_color = (220, 220, 220)
        self.knob_hover_color = (255, 255, 255)
        self.text_color = (255, 255, 255)
        self.label_color = (180, 180, 180)
    
    @property
    def value(self) -> float:
        return self._value
    
    @value.setter
    def value(self, val: float):
        old_val = self._value
        self._value = max(self.min_val, min(self.max_val, val))
        if old_val != self._value and self.on_change:
            self.on_change(self._value)
    
    def _value_to_x(self, val: float) -> int:
        """Convert value to x position."""
        ratio = (val - self.min_val) / (self.max_val - self.min_val) if self.max_val > self.min_val else 0
        return int(self.x + ratio * self.width)
    
    def _x_to_value(self, px: int) -> float:
        """Convert x position to value."""
        ratio = (px - self.x) / self.width if self.width > 0 else 0
        ratio = max(0, min(1, ratio))
        return self.min_val + ratio * (self.max_val - self.min_val)
    
    def get_track_rect(self) -> Tuple[int, int, int, int]:
        """Get the track rectangle (x, y, w, h)."""
        track_height = 6
        track_y = self.y + (self.height - track_height) // 2
        return (self.x, track_y, self.width, track_height)
    
    def get_knob_rect(self) -> Tuple[int, int, int, int]:
        """Get the knob rectangle."""
        knob_width = 12
        knob_height = self.height
        knob_x = self._value_to_x(self._value) - knob_width // 2
        return (knob_x, self.y, knob_width, knob_height)
    
    def point_in_slider(self, px: int, py: int) -> bool:
        """Check if point is in the slider area (track or knob)."""
        # Expand hit area slightly for easier interaction
        expand = 5
        return (self.x - expand <= px <= self.x + self.width + expand and 
                self.y - expand <= py <= self.y + self.height + expand)
    
    def handle_mouse_down(self, px: int, py: int) -> bool:
        """Handle mouse down. Returns True if handled."""
        if self.point_in_slider(px, py):
            self.is_dragging = True
            self.value = self._x_to_value(px)
            return True
        return False
    
    def handle_mouse_up(self, px: int, py: int):
        """Handle mouse up."""
        self.is_dragging = False
    
    def handle_mouse_move(self, px: int, py: int):
        """Handle mouse move."""
        self.hover = self.point_in_slider(px, py)
        if self.is_dragging:
            self.value = self._x_to_value(px)
    
    def draw(self, canvas: np.ndarray):
        """Draw the slider on the canvas."""
        # Draw label
        label_text = f"{self.label}:"
        cv2.putText(canvas, label_text, (self.x - 5, self.y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.label_color, 1)
        
        # Draw track background
        tx, ty, tw, th = self.get_track_rect()
        cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), self.track_color, -1)
        
        # Draw filled portion
        fill_width = self._value_to_x(self._value) - self.x
        fill_color = self.fill_hover_color if (self.hover or self.is_dragging) else self.fill_color
        if fill_width > 0:
            cv2.rectangle(canvas, (tx, ty), (tx + fill_width, ty + th), fill_color, -1)
        
        # Draw track border
        cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (100, 100, 100), 1)
        
        # Draw knob
        kx, ky, kw, kh = self.get_knob_rect()
        knob_color = self.knob_hover_color if (self.hover or self.is_dragging) else self.knob_color
        cv2.rectangle(canvas, (kx, ky), (kx + kw, ky + kh), knob_color, -1)
        cv2.rectangle(canvas, (kx, ky), (kx + kw, ky + kh), (150, 150, 150), 1)
        
        # Draw value text
        value_text = self.format_str.format(self._value)
        text_x = self.x + self.width + 8
        text_y = self.y + self.height // 2 + 4
        cv2.putText(canvas, value_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)


# =============================================================================
# Dropdown Menu Component
# =============================================================================

class DropdownMenu:
    """A dropdown menu component for OpenCV UI."""
    
    def __init__(
        self,
        label: str,
        options: List[str],
        x: int,
        y: int,
        width: int = 180,
        height: int = 25,
        on_change: Optional[Callable[[str, int], None]] = None,
    ):
        self.label = label
        self.options = options
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.on_change = on_change
        
        self.selected_index = 0
        self.is_open = False
        self.hover_index = -1
        
        # Colors
        self.bg_color = (60, 60, 60)
        self.bg_hover_color = (80, 80, 80)
        self.border_color = (120, 120, 120)
        self.text_color = (255, 255, 255)
        self.dropdown_bg = (50, 50, 50)
        self.dropdown_hover = (80, 120, 180)
        self.arrow_color = (200, 200, 200)
    
    @property
    def selected_value(self) -> str:
        return self.options[self.selected_index] if self.options else ""
    
    def set_selected(self, index: int, trigger_callback: bool = False):
        """Set selected index programmatically."""
        if 0 <= index < len(self.options):
            old_index = self.selected_index
            self.selected_index = index
            if trigger_callback and old_index != index and self.on_change:
                self.on_change(self.selected_value, index)
    
    def set_selected_by_value(self, value: str, trigger_callback: bool = False):
        """Set selected by value."""
        if value in self.options:
            self.set_selected(self.options.index(value), trigger_callback)
    
    def update_options(self, options: List[str], keep_selection: bool = False):
        """Update the options list."""
        old_value = self.selected_value if keep_selection else None
        self.options = options
        if keep_selection and old_value in options:
            self.selected_index = options.index(old_value)
        else:
            self.selected_index = 0
    
    def get_main_rect(self) -> Tuple[int, int, int, int]:
        """Get the main button rectangle (x, y, w, h)."""
        return (self.x, self.y, self.width, self.height)
    
    def get_dropdown_rect(self) -> Tuple[int, int, int, int]:
        """Get the dropdown list rectangle when open."""
        dropdown_height = len(self.options) * self.height
        return (self.x, self.y + self.height, self.width, dropdown_height)
    
    def point_in_main(self, px: int, py: int) -> bool:
        """Check if point is in main button area."""
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)
    
    def point_in_dropdown(self, px: int, py: int) -> int:
        """Check if point is in dropdown area. Returns option index or -1."""
        if not self.is_open:
            return -1
        
        dx, dy, dw, dh = self.get_dropdown_rect()
        if dx <= px <= dx + dw and dy <= py <= dy + dh:
            relative_y = py - dy
            return min(relative_y // self.height, len(self.options) - 1)
        return -1
    
    def handle_click(self, px: int, py: int) -> bool:
        """Handle mouse click. Returns True if click was handled."""
        # Check main button
        if self.point_in_main(px, py):
            self.is_open = not self.is_open
            return True
        
        # Check dropdown options
        option_idx = self.point_in_dropdown(px, py)
        if option_idx >= 0:
            old_index = self.selected_index
            self.selected_index = option_idx
            self.is_open = False
            if old_index != option_idx and self.on_change:
                self.on_change(self.selected_value, option_idx)
            return True
        
        # Click outside - close dropdown
        if self.is_open:
            self.is_open = False
            return True
        
        return False
    
    def handle_hover(self, px: int, py: int):
        """Update hover state."""
        self.hover_index = self.point_in_dropdown(px, py)
    
    def draw(self, canvas: np.ndarray):
        """Draw the dropdown on the canvas."""
        # Draw label
        label_width = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.putText(canvas, self.label, (self.x - label_width - 8, self.y + 17),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Draw main button background
        cv2.rectangle(canvas, 
                     (self.x, self.y), 
                     (self.x + self.width, self.y + self.height),
                     self.bg_color, -1)
        
        # Draw border
        border_col = (180, 180, 180) if self.is_open else self.border_color
        cv2.rectangle(canvas,
                     (self.x, self.y),
                     (self.x + self.width, self.y + self.height),
                     border_col, 1)
        
        # Draw selected text (truncated if needed)
        text = self.selected_value
        max_text_width = self.width - 25
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        if text_size[0] > max_text_width:
            while len(text) > 3 and cv2.getTextSize(text + "...", cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] > max_text_width:
                text = text[:-1]
            text = text + "..."
        
        cv2.putText(canvas, text, (self.x + 8, self.y + 17),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.text_color, 1)
        
        # Draw dropdown arrow
        arrow_x = self.x + self.width - 18
        arrow_y = self.y + self.height // 2
        if self.is_open:
            # Up arrow
            pts = np.array([[arrow_x, arrow_y + 3], [arrow_x + 8, arrow_y + 3], [arrow_x + 4, arrow_y - 3]], np.int32)
        else:
            # Down arrow
            pts = np.array([[arrow_x, arrow_y - 3], [arrow_x + 8, arrow_y - 3], [arrow_x + 4, arrow_y + 3]], np.int32)
        cv2.fillPoly(canvas, [pts], self.arrow_color)
        
        # Draw dropdown list if open
        if self.is_open:
            dx, dy, dw, dh = self.get_dropdown_rect()
            
            # Draw dropdown background with shadow effect
            shadow_offset = 3
            cv2.rectangle(canvas,
                         (dx + shadow_offset, dy + shadow_offset),
                         (dx + dw + shadow_offset, dy + dh + shadow_offset),
                         (30, 30, 30), -1)
            
            cv2.rectangle(canvas, (dx, dy), (dx + dw, dy + dh), self.dropdown_bg, -1)
            cv2.rectangle(canvas, (dx, dy), (dx + dw, dy + dh), self.border_color, 1)
            
            # Draw options
            for i, option in enumerate(self.options):
                opt_y = dy + i * self.height
                
                # Highlight on hover or selected
                if i == self.hover_index:
                    cv2.rectangle(canvas, (dx + 1, opt_y + 1), (dx + dw - 1, opt_y + self.height - 1),
                                 self.dropdown_hover, -1)
                elif i == self.selected_index:
                    cv2.rectangle(canvas, (dx + 1, opt_y + 1), (dx + dw - 1, opt_y + self.height - 1),
                                 (70, 70, 70), -1)
                
                # Draw option text
                opt_text = option
                text_size = cv2.getTextSize(opt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                if text_size[0] > dw - 16:
                    while len(opt_text) > 3 and cv2.getTextSize(opt_text + "...", cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] > dw - 16:
                        opt_text = opt_text[:-1]
                    opt_text = opt_text + "..."
                
                cv2.putText(canvas, opt_text, (dx + 8, opt_y + 17),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.text_color, 1)


# =============================================================================
# Live Display Manager with Dropdowns and Sliders
# =============================================================================

class LiveDisplayManager:
    """Manages live display of robot trials using OpenCV with interactive buttons, dropdowns, and sliders."""
    
    BUTTON_HEIGHT = 35
    BUTTON_MARGIN = 10
    TOP_PANEL_HEIGHT = 95  # Increased for dropdown menus AND sliders
    
    def __init__(
        self,
        window_name: str = "SafeLIBERO Live",
        scale: float = 2.0,
        target_fps: int = 20,
        enabled: bool = True,
        task_suites: List[str] = None,
        initial_task_suite: str = None,
        initial_task_id: int = 0,
        initial_safety_margin: float = 0.01,
        initial_guidance_scale: float = 1.0,
    ):
        self.window_name = window_name
        self.scale = scale
        self.target_fps = target_fps
        self.frame_delay = max(1, int(1000 / target_fps))
        self.enabled = enabled and HAS_CV2
        self.paused = False
        self.skip_episode = False
        self.quit_requested = False
        self.restart_requested = False
        self.skip_as_failure = False
        self.screenshot_count = 0
        self.last_frame_time = 0
        
        # Task change requests - these are SET by the UI and READ/CLEARED by the main loop
        self._pending_task_suite: Optional[str] = None
        self._pending_task_id: Optional[int] = None
        self._task_change_requested = False
        
        # Current display values (for showing in UI)
        self._current_task_suite = initial_task_suite or (task_suites[0] if task_suites else "")
        self._current_task_id = initial_task_id
        
        # Safety parameters - these can be read by the main loop at any time
        self._safety_margin = initial_safety_margin
        self._guidance_scale = initial_guidance_scale
        
        # Button state
        self.buttons = {}
        self.button_hover = None
        self.mouse_pos = (0, 0)
        self.video_offset_y = 0
        self.canvas_height = 0
        self.canvas_width = 0
        
        # Initialize UI components
        self.dropdowns: List[DropdownMenu] = []
        self.sliders: List[Slider] = []
        self.task_suite_dropdown = None
        self.task_id_dropdown = None
        self.safety_margin_slider = None
        self.guidance_scale_slider = None
        
        if self.enabled:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            
            # Create task suite dropdown
            task_suites = task_suites or TASK_SUITES
            self.task_suite_dropdown = DropdownMenu(
                label="Suite:",
                options=task_suites,
                x=70,
                y=8,
                width=160,
                height=24,
                on_change=self._on_task_suite_selected,
            )
            if initial_task_suite and initial_task_suite in task_suites:
                self.task_suite_dropdown.set_selected_by_value(initial_task_suite)
            self.dropdowns.append(self.task_suite_dropdown)
            
            # Create task ID dropdown (will be populated based on task suite)
            self.task_id_dropdown = DropdownMenu(
                label="Task:",
                options=[str(i) for i in range(1, 11)],  # Default 1-10
                x=310,
                y=8,
                width=70,
                height=24,
                on_change=self._on_task_id_selected,
            )
            if initial_task_id:
                self.task_id_dropdown.set_selected_by_value(str(initial_task_id))
            self.dropdowns.append(self.task_id_dropdown)
            
            # Create safety margin slider
            self.safety_margin_slider = Slider(
                label="Safety Margin",
                min_val=0.001,
                max_val=0.15,
                initial_val=initial_safety_margin,
                x=120,
                y=45,
                width=120,
                height=18,
                format_str="{:.3f}",
                on_change=self._on_safety_margin_changed,
            )
            self.sliders.append(self.safety_margin_slider)
            
            # Create guidance scale slider
            self.guidance_scale_slider = Slider(
                label="Guidance Scale",
                min_val=0.0,
                max_val=20.0,
                initial_val=initial_guidance_scale,
                x=120,
                y=70,
                width=120,
                height=18,
                format_str="{:.1f}",
                on_change=self._on_guidance_scale_changed,
            )
            self.sliders.append(self.guidance_scale_slider)
            
            logging.info(f"Live display enabled: {self.window_name}")
    
    def _on_task_suite_selected(self, value: str, index: int):
        """Called when task suite is selected from dropdown."""
        logging.info(f"Task suite dropdown selected: {value}")
        self._pending_task_suite = value
        self._task_change_requested = True
    
    def _on_task_id_selected(self, value: str, index: int):
        """Called when task ID is selected from dropdown."""
        task_id = int(value)
        logging.info(f"Task ID dropdown selected: {task_id}")
        self._pending_task_id = task_id
        self._task_change_requested = True
    
    def _on_safety_margin_changed(self, value: float):
        """Called when safety margin slider is changed."""
        self._safety_margin = value
        logging.info(f"Safety margin changed to: {value:.4f}")
    
    def _on_guidance_scale_changed(self, value: float):
        """Called when guidance scale slider is changed."""
        self._guidance_scale = value
        logging.info(f"Guidance scale changed to: {value:.2f}")
    
    @property
    def safety_margin(self) -> float:
        """Get current safety margin value."""
        return self._safety_margin
    
    @property
    def guidance_scale(self) -> float:
        """Get current guidance scale value."""
        return self._guidance_scale
    
    def set_safety_margin(self, value: float):
        """Set safety margin programmatically (also updates slider)."""
        self._safety_margin = value
        if self.safety_margin_slider:
            self.safety_margin_slider._value = value
    
    def set_guidance_scale(self, value: float):
        """Set guidance scale programmatically (also updates slider)."""
        self._guidance_scale = value
        if self.guidance_scale_slider:
            self.guidance_scale_slider._value = value
    
    def update_task_id_options(self, task_ids: List[int]):
        """Update the task ID dropdown options."""
        if self.task_id_dropdown:
            self.task_id_dropdown.update_options([str(i) for i in task_ids], keep_selection=True)
    
    def set_current_task(self, task_suite: str, task_id: int):
        """Update the display to show current task (called by main loop after switching)."""
        self._current_task_suite = task_suite
        self._current_task_id = task_id
        # Update dropdown displays without triggering callbacks
        if self.task_suite_dropdown:
            self.task_suite_dropdown.set_selected_by_value(task_suite, trigger_callback=False)
        if self.task_id_dropdown:
            self.task_id_dropdown.set_selected_by_value(str(task_id), trigger_callback=False)
    
    def has_pending_task_change(self) -> bool:
        """Check if there's a pending task change request."""
        return self._task_change_requested
    
    def get_and_clear_pending_task_change(self) -> Tuple[Optional[str], Optional[int]]:
        """
        Get the pending task change request and clear it.
        Returns (task_suite, task_id) - either may be None if not changed.
        """
        if not self._task_change_requested:
            return None, None
        
        suite = self._pending_task_suite
        task_id = self._pending_task_id
        
        # Clear the pending request
        self._pending_task_suite = None
        self._pending_task_id = None
        self._task_change_requested = False
        
        return suite, task_id
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for button clicks, dropdowns, and sliders."""
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # First check sliders (they take priority for dragging)
            slider_handled = False
            for slider in self.sliders:
                if slider.handle_mouse_down(x, y):
                    slider_handled = True
                    break
            
            if not slider_handled:
                # Then check dropdowns
                dropdown_handled = False
                for dropdown in self.dropdowns:
                    if dropdown.handle_click(x, y):
                        dropdown_handled = True
                        break
                
                if not dropdown_handled:
                    # Close any open dropdowns
                    for dropdown in self.dropdowns:
                        dropdown.is_open = False
                    # Then check buttons
                    self._handle_click(self.mouse_pos)
        
        elif event == cv2.EVENT_LBUTTONUP:
            # Release slider dragging
            for slider in self.sliders:
                slider.handle_mouse_up(x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            self._update_hover(self.mouse_pos)
            # Update dropdown hover states
            for dropdown in self.dropdowns:
                dropdown.handle_hover(x, y)
            # Update slider hover states and dragging
            for slider in self.sliders:
                slider.handle_mouse_move(x, y)
    
    def _handle_click(self, pos):
        """Handle mouse click on buttons."""
        for name, btn in self.buttons.items():
            if self._point_in_rect(pos, btn['rect']):
                self._trigger_button(name)
                return
    
    def _update_hover(self, pos):
        """Update button hover state."""
        self.button_hover = None
        for name, btn in self.buttons.items():
            if self._point_in_rect(pos, btn['rect']):
                self.button_hover = name
                return
    
    def _point_in_rect(self, point, rect):
        """Check if point is inside rectangle."""
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh
    
    def _trigger_button(self, button_name):
        """Trigger button action."""
        if button_name == 'restart':
            self.restart_requested = True
            logging.info("Restart requested via button")
        elif button_name == 'skip_fail':
            self.skip_as_failure = True
            logging.info("Skip as failure requested via button")
        elif button_name == 'pause':
            self.paused = not self.paused
        elif button_name == 'next':
            self.skip_episode = True
        elif button_name == 'quit':
            self.quit_requested = True
    
    def show_frame(
        self,
        frame: np.ndarray,
        episode_info: Optional[dict] = None,
        force_pause: bool = False,
        wrist_frame: Optional[np.ndarray] = None,
    ) -> str:
        """
        Display a frame and handle user input.
        
        Args:
            frame: RGB image (uint8) - main agentview camera
            episode_info: Optional dict with episode metadata to overlay
            force_pause: Force pause on this frame (e.g., collision detected)
            wrist_frame: Optional RGB image (uint8) - wrist camera view
            
        Returns:
            Action string: 'continue', 'skip', 'quit', 'restart', 'skip_fail', 
                          'task_change', or 'screenshot'
        """
        if not self.enabled:
            return 'continue'
        
        # Check for pending task change FIRST (highest priority)
        if self._task_change_requested:
            return 'task_change'
        
        # Check for pending actions from button clicks
        if self.restart_requested:
            self.restart_requested = False
            return 'restart'
        if self.skip_as_failure:
            self.skip_as_failure = False
            return 'skip_fail'
        
        # Convert RGB to BGR for OpenCV
        video_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Scale the video frame FIRST
        if self.scale != 1.0:
            h, w = video_frame.shape[:2]
            new_w, new_h = int(w * self.scale), int(h * self.scale)
            video_frame = cv2.resize(video_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        video_h, video_w = video_frame.shape[:2]
        
        # Process wrist frame if provided
        if wrist_frame is not None:
            wrist_bgr = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)
            wrist_h, wrist_w = wrist_bgr.shape[:2]
            wrist_scale = video_h / wrist_h
            new_wrist_w = int(wrist_w * wrist_scale)
            wrist_bgr = cv2.resize(wrist_bgr, (new_wrist_w, video_h), interpolation=cv2.INTER_LINEAR)
            
            cv2.putText(wrist_bgr, "WRIST", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            separator_width = 4
            separator = np.zeros((video_h, separator_width, 3), dtype=np.uint8)
            separator[:] = (80, 80, 80)
            
            combined_frame = np.hstack([video_frame, separator, wrist_bgr])
            video_w = combined_frame.shape[1]
            video_frame = combined_frame
        
        # Create canvas with space for UI elements
        top_bar_height = self.TOP_PANEL_HEIGHT
        button_panel_height = 50
        
        total_height = video_h + top_bar_height + button_panel_height
        canvas = np.zeros((total_height, video_w, 3), dtype=np.uint8)
        canvas[:] = (50, 50, 50)
        
        # Place the scaled video frame
        canvas[top_bar_height:top_bar_height + video_h, :] = video_frame
        
        # Draw top panel with dropdowns and sliders
        self._draw_top_panel(canvas, video_w, top_bar_height, episode_info)
        
        # Draw bottom button panel
        button_panel_y = top_bar_height + video_h
        self._draw_button_panel(canvas, video_w, button_panel_y, button_panel_height)
        
        # Store geometry
        self.video_offset_y = top_bar_height
        self.canvas_width = video_w
        self.canvas_height = total_height
        
        # Show the frame
        cv2.imshow(self.window_name, canvas)
        
        # Handle pause state
        if force_pause or self.paused:
            return self._handle_pause_loop(canvas)
        
        # Non-blocking key check with frame delay
        key = cv2.waitKey(self.frame_delay) & 0xFF
        return self._handle_key(key, canvas)
    
    def _draw_top_panel(self, canvas: np.ndarray, width: int, height: int, info: Optional[dict]):
        """Draw the top panel with dropdowns, sliders, and episode info."""
        # Draw dropdowns
        for dropdown in self.dropdowns:
            dropdown.draw(canvas)
        
        # Draw sliders
        for slider in self.sliders:
            slider.draw(canvas)
        
        # Draw "TASK CHANGE PENDING" indicator if applicable
        if self._task_change_requested:
            cv2.putText(canvas, "SWITCHING TASK...", (width // 2 - 80, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw episode info on the right side
        if info is not None:
            task_id = info.get('task_id', '?')
            episode = info.get('episode', '?')
            step = info.get('step', '?')
            max_steps = info.get('max_steps', '?')
            collision = info.get('collision', False)
            
            # Episode info text - positioned on right side of top panel
            info_text = f"Ep {episode} | Step {step}/{max_steps}"
            text_width = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            cv2.putText(canvas, info_text, (width - text_width - 15, 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Current safety params display
            safety_text = f"M:{self._safety_margin:.3f} G:{self._guidance_scale:.1f}"
            cv2.putText(canvas, safety_text, (width - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 255), 1)
            
            # Collision indicator
            if collision:
                cv2.putText(canvas, "COLLISION", (width - 100, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Pause indicator (only show if not switching tasks)
            if self.paused and not self._task_change_requested:
                cv2.putText(canvas, "PAUSED", (width // 2 - 40, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw separator line
        cv2.line(canvas, (0, height - 1), (width, height - 1), (80, 80, 80), 1)
    
    def _draw_button_panel(self, canvas: np.ndarray, width: int, panel_y: int, panel_height: int):
        """Draw the bottom button panel."""
        button_width = 100
        button_height = self.BUTTON_HEIGHT - 10
        button_y = panel_y + self.BUTTON_MARGIN
        
        buttons_config = [
            ('restart', 'RESTART', (80, 180, 255), (60, 140, 200)),
            ('skip_fail', 'SKIP+FAIL', (80, 80, 255), (60, 60, 200)),
            ('pause', 'PAUSE' if not self.paused else 'RESUME', (255, 200, 80), (200, 160, 60)),
            ('next', 'NEXT', (180, 180, 80), (140, 140, 60)),
            ('quit', 'QUIT', (100, 100, 100), (80, 80, 80)),
        ]
        
        total_width = len(buttons_config) * (button_width + self.BUTTON_MARGIN) - self.BUTTON_MARGIN
        start_x = (width - total_width) // 2
        
        self.buttons = {}
        
        for i, (name, label, color, hover_color) in enumerate(buttons_config):
            btn_x = start_x + i * (button_width + self.BUTTON_MARGIN)
            rect = (btn_x, button_y, button_width, button_height)
            
            self.buttons[name] = {'rect': rect, 'label': label}
            
            btn_color = hover_color if self.button_hover == name else color
            
            cv2.rectangle(canvas, 
                         (btn_x, button_y), 
                         (btn_x + button_width, button_y + button_height),
                         btn_color, -1)
            
            border_color = (255, 255, 255) if self.button_hover == name else (150, 150, 150)
            thickness = 2 if self.button_hover == name else 1
            cv2.rectangle(canvas,
                         (btn_x, button_y),
                         (btn_x + button_width, button_y + button_height),
                         border_color, thickness)
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            text_x = btn_x + (button_width - text_size[0]) // 2
            text_y = button_y + (button_height + text_size[1]) // 2
            cv2.putText(canvas, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        hint_text = "Keys: [R]estart [F]ail [SPACE]Pause [N]ext [S]creenshot [Q]uit | Drag sliders to adjust safety params"
        cv2.putText(canvas, hint_text, (10, panel_y + panel_height - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1)
    
    def _handle_pause_loop(self, display_frame: np.ndarray) -> str:
        """Handle input during pause state."""
        while True:
            # Check for task change during pause
            if self._task_change_requested:
                self.paused = False
                return 'task_change'
            
            if self.restart_requested:
                self.restart_requested = False
                self.paused = False
                return 'restart'
            if self.skip_as_failure:
                self.skip_as_failure = False
                self.paused = False
                return 'skip_fail'
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(100) & 0xFF
            action = self._handle_key(key, display_frame)
            
            if action != 'continue' or not self.paused:
                return action
    
    def _handle_key(self, key: int, frame: np.ndarray) -> str:
        """Handle keyboard input."""
        if key == ord('q') or key == 27:
            self.quit_requested = True
            return 'quit'
        elif key == ord(' '):
            self.paused = not self.paused
            return 'continue'
        elif key == ord('n'):
            self.skip_episode = True
            return 'skip'
        elif key == ord('r'):
            return 'restart'
        elif key == ord('f'):
            return 'skip_fail'
        elif key == ord('s'):
            self._save_screenshot(frame)
            return 'screenshot'
        
        return 'continue'
    
    def _save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot."""
        self.screenshot_count += 1
        filename = f"screenshot_{self.screenshot_count:04d}.png"
        cv2.imwrite(filename, frame)
        logging.info(f"Screenshot saved: {filename}")
    
    def reset_episode_flags(self):
        """Reset per-episode flags."""
        self.skip_episode = False
        self.restart_requested = False
        self.skip_as_failure = False
    
    def close(self):
        """Close the display window."""
        if self.enabled:
            cv2.destroyAllWindows()
    
    def is_quit_requested(self) -> bool:
        """Check if user requested to quit."""
        return self.quit_requested


# =============================================================================
# Obstacle Visualization Overlay
# =============================================================================

def draw_safety_overlay(
    image: np.ndarray,
    eef_pos: np.ndarray,
    obs_info,
    safety_margin: float = 0.08,
    collision: bool = False,
) -> np.ndarray:
    """Draw safety visualization overlay on image."""
    img = image.copy()
    h, w = img.shape[:2]
    
    bar_height = 35
    img[:bar_height, :] = (40, 40, 40)
    
    if obs_info is None:
        cv2.putText(img, "NO OBSTACLE", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        return img
    
    distance = obs_info.distance
    obs_pos = obs_info.position if hasattr(obs_info, 'position') else obs_info.center
    surface_pt = obs_info.surface_point
    
    if collision:
        status, color = "COLLISION", (255, 50, 50)
    elif distance < 0:
        status, color = "INSIDE!", (255, 0, 255)
    elif distance < safety_margin * 0.5:
        status, color = "DANGER", (255, 80, 80)
    elif distance < safety_margin:
        status, color = "CAUTION", (255, 200, 50)
    else:
        status, color = "SAFE", (50, 255, 50)
    
    cv2.putText(img, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(img, f"Dist: {distance*100:.1f}cm", (130, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Margin: {safety_margin*100:.0f}cm", (250, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    bar_x = 10
    bar_y = h - 25
    bar_width = w - 120
    bar_h = 12
    
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_h), (50, 50, 50), -1)
    
    max_dist = safety_margin * 3
    margin_x = int(bar_x + (safety_margin / max_dist) * bar_width)
    cv2.line(img, (margin_x, bar_y - 2), (margin_x, bar_y + bar_h + 2), (50, 200, 255), 2)
    
    dist_ratio = min(max(distance, 0) / max_dist, 1.0)
    dist_x = int(bar_x + dist_ratio * bar_width)
    cv2.rectangle(img, (bar_x, bar_y), (dist_x, bar_y + bar_h), color, -1)
    
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_h), (100, 100, 100), 1)
    
    if collision:
        border = 8
        cv2.rectangle(img, (0, 0), (w-1, h-1), (255, 50, 50), border)
    
    return img


# =============================================================================
# Obstacle Tracking
# =============================================================================

@dataclass
class ObstacleInfo:
    """Obstacle information for safety guidance."""
    name: str
    center: np.ndarray
    radii: np.ndarray
    surface_point: np.ndarray
    distance: float


@dataclass
class GroundTruthObstacleInfo:
    """Ground truth obstacle information."""
    name: str
    geom_type: int
    position: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    surface_point: np.ndarray
    distance: float


class GroundTruthObstacleTracker:
    """Tracks obstacles using MuJoCo's ground truth geometry."""
    
    PLANE = 0
    HFIELD = 1
    SPHERE = 2
    CAPSULE = 3
    ELLIPSOID = 4
    CYLINDER = 5
    BOX = 6
    MESH = 7
    
    def __init__(self, env, verbose: bool = False):
        self.env = env
        self.verbose = verbose
        self.obstacle_geoms: Dict[str, Dict[str, Any]] = {}
        self._find_obstacle_geoms()
    
    def _find_obstacle_geoms(self):
        """Find all obstacle geoms in the scene."""
        sim = self.env.sim
        model = sim.model
        
        for geom_id in range(model.ngeom):
            geom_name = model.geom_id2name(geom_id)
            if geom_name and "obstacle" in geom_name.lower():
                geom_type = model.geom_type[geom_id]
                geom_size = model.geom_size[geom_id].copy()
                
                self.obstacle_geoms[geom_name] = {
                    'geom_id': geom_id,
                    'geom_type': geom_type,
                    'size': geom_size,
                }
    
    def get_geom_state(self, geom_name: str):
        """Get current position and rotation of a geom."""
        sim = self.env.sim
        geom_id = self.obstacle_geoms[geom_name]['geom_id']
        
        pos = sim.data.geom_xpos[geom_id].copy()
        rot = sim.data.geom_xmat[geom_id].reshape(3, 3).copy()
        
        return pos, rot
    
    def closest_point_on_box(self, point, box_pos, box_rot, half_extents):
        """Find closest point on box surface to a query point."""
        local_point = box_rot.T @ (point - box_pos)
        clamped = np.clip(local_point, -half_extents, half_extents)
        inside = np.all(np.abs(local_point) <= half_extents)
        
        if inside:
            distances_to_faces = half_extents - np.abs(local_point)
            closest_axis = np.argmin(distances_to_faces)
            clamped[closest_axis] = np.sign(local_point[closest_axis]) * half_extents[closest_axis]
            surface_point_world = box_rot @ clamped + box_pos
            distance = -distances_to_faces[closest_axis]
        else:
            surface_point_world = box_rot @ clamped + box_pos
            distance = np.linalg.norm(point - surface_point_world)
        
        return surface_point_world, distance
    
    def closest_point_on_sphere(self, point, sphere_pos, radius):
        """Find closest point on sphere surface."""
        direction = point - sphere_pos
        dist_to_center = np.linalg.norm(direction)
        
        if dist_to_center < 1e-8:
            surface_point = sphere_pos + np.array([radius, 0, 0])
            distance = -radius
        else:
            direction_normalized = direction / dist_to_center
            surface_point = sphere_pos + radius * direction_normalized
            distance = dist_to_center - radius
        
        return surface_point, distance
    
    def closest_point_on_cylinder(self, point, cyl_pos, cyl_rot, radius, half_height):
        """Find closest point on cylinder surface."""
        local_point = cyl_rot.T @ (point - cyl_pos)
        radial = local_point[:2]
        axial = local_point[2]
        radial_dist = np.linalg.norm(radial)
        clamped_axial = np.clip(axial, -half_height, half_height)
        
        inside_radial = radial_dist <= radius
        inside_axial = np.abs(axial) <= half_height
        inside = inside_radial and inside_axial
        
        if inside:
            dist_to_curved = radius - radial_dist
            dist_to_cap = half_height - np.abs(axial)
            
            if dist_to_curved < dist_to_cap:
                if radial_dist > 1e-8:
                    clamped_radial = radial / radial_dist * radius
                else:
                    clamped_radial = np.array([radius, 0])
                local_surface = np.array([clamped_radial[0], clamped_radial[1], axial])
                distance = -dist_to_curved
            else:
                local_surface = np.array([radial[0], radial[1], np.sign(axial) * half_height])
                distance = -dist_to_cap
        else:
            if radial_dist > 1e-8:
                clamped_radial = radial / radial_dist * min(radial_dist, radius)
            else:
                clamped_radial = np.array([0, 0])
            
            local_surface = np.array([clamped_radial[0], clamped_radial[1], clamped_axial])
            surface_point_world = cyl_rot @ local_surface + cyl_pos
            distance = np.linalg.norm(point - surface_point_world)
            return surface_point_world, distance
        
        surface_point_world = cyl_rot @ local_surface + cyl_pos
        return surface_point_world, distance
    
    def get_obstacle_info(self, eef_pos: np.ndarray) -> Optional[GroundTruthObstacleInfo]:
        """Get ground truth obstacle info for the closest obstacle to EEF."""
        if not self.obstacle_geoms:
            return None
        
        best_info = None
        best_distance = float('inf')
        
        for geom_name, geom_data in self.obstacle_geoms.items():
            pos, rot = self.get_geom_state(geom_name)
            geom_type = geom_data['geom_type']
            size = geom_data['size']
            
            if geom_type == self.BOX:
                surface_point, distance = self.closest_point_on_box(eef_pos, pos, rot, size[:3])
            elif geom_type == self.SPHERE:
                surface_point, distance = self.closest_point_on_sphere(eef_pos, pos, size[0])
            elif geom_type == self.CYLINDER:
                surface_point, distance = self.closest_point_on_cylinder(eef_pos, pos, rot, size[0], size[1])
            elif geom_type == self.MESH:
                surface_point, distance = self.closest_point_on_box(eef_pos, pos, rot, size[:3])
            else:
                continue
            
            if distance < best_distance:
                best_distance = distance
                best_info = GroundTruthObstacleInfo(
                    name=geom_name,
                    geom_type=geom_type,
                    position=pos,
                    size=size,
                    rotation=rot,
                    surface_point=surface_point,
                    distance=distance,
                )
        
        return best_info


# =============================================================================
# Trajectory Visualizer
# =============================================================================

class TrajectoryVisualizer:
    def __init__(self, env, camera_name="agentview", width=512, height=512, safety_margin=0.05):
        self.env = env
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.safety_margin = safety_margin
        self.predicted_trajectory = None
        self.history_trajectory = None
        self.overlay_trajectories = []
        self.overlay_alpha = 0.55
        self.obstacle_info = None

    def set_predicted_trajectory(self, trajectory):
        """Set predicted trajectory (N, 3) world positions."""
        self.predicted_trajectory = trajectory

    def set_overlay_trajectories(self, overlays, alpha=0.55):
        """Prior-condition trajectories to draw semi-transparently.
        overlays: list of {'positions': (N,3) world coords, 'color': (R,G,B)}."""
        self.overlay_trajectories = overlays or []
        self.overlay_alpha = alpha

    def set_history_trajectory(self, trajectory):
        """Set the actual EEF path so far (list/array of world positions)."""
        self.history_trajectory = trajectory

    def set_obstacle_info(self, obs_info):
        self.obstacle_info = obs_info
    def draw_obstacle_bbox(self, img, obstacle, color=(0, 255, 255), thickness=1):
        """Draw wireframe bounding box of an ObstacleGeometry on the image."""
        img = np.ascontiguousarray(img)
        
        pos = obstacle.position
        rot = obstacle.rotation
        half = obstacle.size
        
        corners_local = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    corners_local.append(np.array([sx * half[0], sy * half[1], sz * half[2]]))
        
        corners_world = [rot @ c + pos for c in corners_local]
        corners_px = [self.world_to_pixel(c) for c in corners_world]
        
        edges = [
            (0,1), (2,3), (4,5), (6,7),
            (0,2), (1,3), (4,6), (5,7),
            (0,4), (1,5), (2,6), (3,7),
        ]
        
        for i, j in edges:
            if corners_px[i] is not None and corners_px[j] is not None:
                cv2.line(img, corners_px[i], corners_px[j], color, thickness)
        
        center_px = self.world_to_pixel(pos)
        if center_px is not None:
            cv2.circle(img, center_px, 4, color, -1)
        
        return img
    def render_frame(self, obstacles=None, eef_pos=None, eef_radii=None):
        """Render frame with trajectory overlay and 3D ellipsoids."""
        import mujoco
        
        sim = self.env.sim
        ctx = sim._render_context_offscreen
        
        if ctx is None or not (obstacles or (eef_pos is not None and eef_radii is not None)):
            img = sim.render(
                width=self.width,
                height=self.height,
                camera_name=self.camera_name
            )
        else:
            cam_id = sim.model.camera_name2id(self.camera_name)
            
            # Create a camera struct pointed at the fixed camera
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = cam_id
            
            # Update scene with the correct camera
            mujoco.mjv_updateScene(
                sim.model._model,
                sim.data._data,
                ctx.vopt,
                ctx.pert,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                ctx.scn,
            )
            
            # Inject ellipsoids
            self._inject_ellipsoids(ctx.scn, obstacles, eef_pos, eef_radii)
            
            # Render
            ctx.update_offscreen_size(self.width, self.height)
            mujoco.mjr_render(
                mujoco.MjrRect(0, 0, self.width, self.height),
                ctx.scn,
                ctx.con,
            )
            
            img = np.empty((self.height, self.width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, mujoco.MjrRect(0, 0, self.width, self.height), ctx.con)
        
        img = img[::-1]
        img = np.ascontiguousarray(img)
        
        # Scale overlay sizes with render resolution (calibrated for 224px).
        s = max(1.0, self.height / 224.0)
        line_th = max(2, int(round(2 * s)))
        hist_th = max(2, int(round(3 * s)))
        hist_dot_r = max(2, int(round(2.5 * s)))
        dot_r = max(3, int(round(4 * s)))
        out_r = max(1, int(round(0.3 * s)))  # thin outline on future dots
        x_sz = max(4, int(round(6 * s)))
        ov_th = max(2, int(round(3 * s)))

        # Prior-condition trajectories, blended at overlay_alpha so they read as
        # partially opaque underlays beneath the live trace.
        if self.overlay_trajectories:
            layer = img.copy()
            for ov in self.overlay_trajectories:
                color = tuple(int(c) for c in ov["color"])
                opx = [self.world_to_pixel(p) for p in ov["positions"]]
                for i in range(len(opx) - 1):
                    if opx[i] is None or opx[i + 1] is None:
                        continue
                    cv2.line(layer, opx[i], opx[i + 1], color, ov_th)
            a = float(self.overlay_alpha)
            img = cv2.addWeighted(layer, a, img, 1.0 - a, 0)

        # Historic EEF path (purple) with depth shading and a dot per executed
        # action: near the camera -> brighter/thicker, far -> dimmer/thinner.
        if self.history_trajectory is not None and len(self.history_trajectory) >= 2:
            base_purple = (160.0, 32.0, 240.0)
            pts, depths = [], []
            for p in self.history_trajectory:
                px, d = self.world_to_pixel_with_depth(p)
                pts.append(px)
                depths.append(d)
            valid_d = [d for d in depths if d is not None]
            d_lo, d_hi = (min(valid_d), max(valid_d)) if valid_d else (0.0, 1.0)
            d_rng = max(d_hi - d_lo, 1e-6)

            def _shade(d):
                near = 1.0 - (d - d_lo) / d_rng  # 1 near camera, 0 far
                bright = 0.4 + 0.6 * near
                color = tuple(int(c * bright) for c in base_purple)
                th = max(1, int(round(hist_th * (0.45 + 0.55 * near))))
                rad = max(2, int(round(hist_dot_r * (0.5 + 0.5 * near))))
                return color, th, rad

            for i in range(len(pts) - 1):
                if pts[i] is None or pts[i + 1] is None or depths[i] is None or depths[i + 1] is None:
                    continue
                color, th, _ = _shade(0.5 * (depths[i] + depths[i + 1]))
                cv2.line(img, pts[i], pts[i + 1], color, th)

            # One dot per executed action, drawn far->near so closer dots occlude
            # farther ones (painter's algorithm => stronger depth cue).
            dot_order = sorted(
                (i for i in range(len(pts)) if pts[i] is not None and depths[i] is not None),
                key=lambda i: depths[i], reverse=True,
            )
            for i in dot_order:
                color, _, rad = _shade(depths[i])
                cv2.circle(img, pts[i], rad, color, -1)
                cv2.circle(img, pts[i], rad, (60, 12, 90), 1)

        if self.predicted_trajectory is not None:
            pixels = []
            violations = []

            for pos in self.predicted_trajectory:
                px = self.world_to_pixel(pos)
                if px is not None:
                    pixels.append(px)
                    is_violation = False
                    if self.obstacle_info is not None:
                        obs_pos = self.obstacle_info.surface_point
                        dist = np.linalg.norm(pos - obs_pos)
                        is_violation = dist < self.safety_margin
                    violations.append(is_violation)
                else:
                    pixels.append(None)
                    violations.append(False)

            for i in range(len(pixels) - 1):
                if pixels[i] is None or pixels[i+1] is None:
                    continue
                t = i / max(len(pixels) - 1, 1)
                color = (int(255 * t), int(255 * (1 - t)), 0)
                cv2.line(img, pixels[i], pixels[i+1], color, line_th)

            for i, (px, is_violation) in enumerate(zip(pixels, violations)):
                if px is None:
                    continue
                t = i / max(len(pixels) - 1, 1)

                if is_violation:
                    self._draw_x(img, px, size=x_sz, color=(0, 0, 255), thickness=line_th)
                else:
                    color = (int(255 * t), int(255 * (1 - t)), 0)
                    cv2.circle(img, px, dot_r, color, -1)
                    cv2.circle(img, px, dot_r, (255, 255, 255), out_r)
        
        if self.obstacle_info is not None and hasattr(self.obstacle_info, 'surface_point'):
            px = self.world_to_pixel(self.obstacle_info.surface_point)
            # if px is not None:
            #     cv2.circle(img, px, 8, (0, 255, 255), 2)
        
        return img[:, ::-1]

    def _inject_ellipsoids(self, scn, obstacles=None, eef_pos=None, eef_radii=None):
        """Inject ellipsoid geoms into the MjvScene."""
        import mujoco
        
        if obstacles:
            for obs in obstacles:
                self._add_scene_geom(
                    scn,
                    pos=obs.position,
                    size=obs.size,
                    mat=obs.rotation,
                    rgba=np.array([0.0, 0.4, 1.0, 0.3], dtype=np.float32),
                )
        
        if eef_pos is not None and eef_radii is not None:
            self._add_scene_geom(
                scn,
                pos=eef_pos,
                size=eef_radii,
                mat=np.eye(3),
                rgba=np.array([0.0, 1.0, 0.0, 0.3], dtype=np.float32),
            )

    def _add_scene_geom(self, scn, pos, size, mat, rgba):
        """Add an ellipsoid geom to the MjvScene."""
        import mujoco
        
        if scn.ngeom >= scn.maxgeom:
            return
        
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=np.array(size, dtype=np.float64),
            pos=np.array(pos, dtype=np.float64),
            mat=np.array(mat, dtype=np.float64).flatten(),
            rgba=np.array(rgba, dtype=np.float32),
        )
        
        scn.ngeom += 1
    def _draw_x(self, img, center, size=6, color=(0, 0, 255), thickness=2):
        """Draw an X marker at the given center point."""
        x, y = center
        cv2.line(img, (x - size, y - size), (x + size, y + size), color, thickness)
        cv2.line(img, (x - size, y + size), (x + size, y - size), color, thickness)

    def world_to_pixel(self, world_pos):
        """Convert world position to pixel coordinates."""
        sim = self.env.sim
        cam_id = sim.model.camera_name2id(self.camera_name)
        cam_pos = sim.data.cam_xpos[cam_id]
        cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
        point_in_cam = cam_mat.T @ (world_pos - cam_pos)
        fovy = sim.model.cam_fovy[cam_id]
        f = self.height / (2 * np.tan(np.radians(fovy) / 2))
        
        if point_in_cam[2] > 0:
            return None
        
        x_norm = point_in_cam[0] / (-point_in_cam[2])
        y_norm = point_in_cam[1] / (-point_in_cam[2])
        x_pixel = x_norm * f + self.width / 2
        y_pixel = self.height / 2 - y_norm * f
        
        if 0 <= x_pixel < self.width and 0 <= y_pixel < self.height:
            return (int(x_pixel), int(y_pixel))
        return None

    def world_to_pixel_with_depth(self, world_pos):
        """Like world_to_pixel but also returns camera-space depth (meters in front)."""
        sim = self.env.sim
        cam_id = sim.model.camera_name2id(self.camera_name)
        cam_pos = sim.data.cam_xpos[cam_id]
        cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
        point_in_cam = cam_mat.T @ (world_pos - cam_pos)
        fovy = sim.model.cam_fovy[cam_id]
        f = self.height / (2 * np.tan(np.radians(fovy) / 2))

        if point_in_cam[2] > 0:
            return None, None

        depth = -point_in_cam[2]
        x_norm = point_in_cam[0] / depth
        y_norm = point_in_cam[1] / depth
        x_pixel = x_norm * f + self.width / 2
        y_pixel = self.height / 2 - y_norm * f

        if 0 <= x_pixel < self.width and 0 <= y_pixel < self.height:
            return (int(x_pixel), int(y_pixel)), float(depth)
        return None, None

    def draw_obstacle_ellipsoid(self, img, obstacle, color=(255, 100, 0), alpha=0.3, n_points=50):
        """Draw a 3D ellipsoid overlay for an obstacle."""
        img = np.ascontiguousarray(img)
        
        pos = obstacle.position
        rot = obstacle.rotation
        half = obstacle.size  # semi-axes
        
        # Generate surface points on unit sphere using lat/lon sampling
        surface_pixels = []
        
        for i in range(n_points):
            lat = np.pi * (i / (n_points - 1)) - np.pi / 2  # -pi/2 to pi/2
            for j in range(n_points):
                lon = 2 * np.pi * (j / (n_points - 1))  # 0 to 2pi
                
                # Unit sphere point
                x = np.cos(lat) * np.cos(lon)
                y = np.cos(lat) * np.sin(lon)
                z = np.sin(lat)
                
                # Scale by semi-axes
                local = np.array([x * half[0], y * half[1], z * half[2]])
                
                # Transform to world
                world = rot @ local + pos
                
                px = self.world_to_pixel(world)
                if px is not None:
                    surface_pixels.append(px)
        
        if len(surface_pixels) < 3:
            return img
        
        # Compute convex hull of projected points
        points_array = np.array(surface_pixels, dtype=np.int32)
        hull = cv2.convexHull(points_array)
        
        # Draw filled semi-transparent ellipsoid
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, hull, color)
        cv2.polylines(overlay, [hull], True, color, 2)
        
        # Blend
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Draw wireframe latitude/longitude lines for 3D effect
        # Longitude lines (vertical slices)
        for j in range(0, n_points, n_points // 8):
            lon = 2 * np.pi * (j / (n_points - 1))
            line_pixels = []
            for i in range(n_points):
                lat = np.pi * (i / (n_points - 1)) - np.pi / 2
                x = np.cos(lat) * np.cos(lon)
                y = np.cos(lat) * np.sin(lon)
                z = np.sin(lat)
                local = np.array([x * half[0], y * half[1], z * half[2]])
                world = rot @ local + pos
                px = self.world_to_pixel(world)
                if px is not None:
                    line_pixels.append(px)
            
            if len(line_pixels) > 1:
                pts = np.array(line_pixels, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
        
        # Latitude lines (horizontal slices)
        for i in range(0, n_points, n_points // 6):
            lat = np.pi * (i / (n_points - 1)) - np.pi / 2
            line_pixels = []
            for j in range(n_points):
                lon = 2 * np.pi * (j / (n_points - 1))
                x = np.cos(lat) * np.cos(lon)
                y = np.cos(lat) * np.sin(lon)
                z = np.sin(lat)
                local = np.array([x * half[0], y * half[1], z * half[2]])
                world = rot @ local + pos
                px = self.world_to_pixel(world)
                if px is not None:
                    line_pixels.append(px)
            
            if len(line_pixels) > 1:
                pts = np.array(line_pixels, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
        
        return img
# =============================================================================
# Policy Client
# =============================================================================

class PredictiveCBFFilter:
    """Predictive CBF filter for trajectory safety."""
    
    def __init__(
        self,
        safety_margin: float = 0.08,
        obstacle_radius: float = 0.02,
        max_iterations: int = 10,
        step_size: float = 0.3,
    ):
        self.safety_margin = safety_margin
        self.obstacle_radius = obstacle_radius
        self.safe_dist_sq = (obstacle_radius + safety_margin) ** 2
        self.max_iterations = max_iterations
        self.step_size = step_size
    
    def update_safety_margin(self, safety_margin: float):
        """Update the safety margin dynamically."""
        self.safety_margin = safety_margin
        self.safe_dist_sq = (self.obstacle_radius + safety_margin) ** 2
    
    def filter(self, actions, eef_pos, obs_pos):
        """Apply predictive CBF filtering."""
        actions = actions.copy()
        info = {'iterations': 0, 'was_modified': False}
        
        for iteration in range(self.max_iterations):
            positions = [eef_pos.copy()]
            pos = eef_pos.copy()
            for action in actions:
                pos = pos + action[:3]
                positions.append(pos.copy())
            
            clearances = [np.sum((p - obs_pos)**2) - self.safe_dist_sq for p in positions]
            t_star = np.argmin(clearances)
            min_clearance = clearances[t_star]
            
            if min_clearance >= 0:
                info['iterations'] = iteration
                break
            
            if t_star == 0:
                break
            
            info['was_modified'] = True
            grad_h = 2.0 * (positions[t_star] - obs_pos)
            grad_norm = np.linalg.norm(grad_h)
            
            if grad_norm < 1e-8:
                break
            
            violation = -min_clearance
            for t in range(min(t_star, len(actions))):
                weight = (t_star - t) / t_star
                correction = self.step_size * weight * violation * grad_h / grad_norm
                actions[t, :3] += correction
            
            info['iterations'] = iteration + 1

        return actions, info


class SingleActionCBF:
    """Single-action Control Barrier Function safety layer (VLSA/AEGIS,
    arXiv:2512.11891).

    Applies a CBF to the action emitted by the (unsafe) VLA model at each control
    step -- no trajectory-level modification. Solves the QP

        u_safe = argmin_u ||u - u_vla||^2   s.t.   h_dot(x) >= -alpha(h(x))

    with linear class-K function alpha(h) = alpha * h and single-integrator
    dynamics p_dot = action_scale * u (paper uses action_scale=0.2, alpha=10).
    Both the obstacle and the end-effector are ellipsoids; the barrier uses the
    obstacle ellipsoid inflated by the EEF ellipsoid radii + safety margin
    (Minkowski-sum approximation), giving

        h(p)      = (p-c)^T M (p-c) - 1,   M = R diag(1/r_i^2) R^T
        grad h(p) = 2 M (p-c)

    Linearizing h(p_{t+1}) with p_{t+1} = p + action_scale*dt*u, the discrete CBF
    constraint reduces (dt cancels) to the single linear inequality

        grad_h . u >= -(alpha / action_scale) * h(p),

    whose min-norm projection has a closed form.
    """

    def __init__(self, alpha=10.0, action_scale=0.2,
                 eef_radii=(0.06, 0.12, 0.11), safety_margin=0.0, eef_scale=1.0):
        self.alpha = float(alpha)
        self.action_scale = float(action_scale)
        self.eef_radii = np.asarray(eef_radii, dtype=float) * float(eef_scale)
        self.safety_margin = float(safety_margin)

    def update_safety_margin(self, safety_margin: float):
        self.safety_margin = float(safety_margin)

    def _barrier(self, p, obstacle):
        c = np.asarray(obstacle.position, dtype=float)
        R = np.asarray(obstacle.rotation, dtype=float).reshape(3, 3)
        radii = np.asarray(obstacle.size, dtype=float)
        r = radii + self.eef_radii + self.safety_margin  # Minkowski inflation
        M = R @ np.diag(1.0 / (r ** 2)) @ R.T
        d = np.asarray(p, dtype=float) - c
        h = float(d @ M @ d) - 1.0
        grad = 2.0 * (M @ d)
        return h, grad

    def filter(self, action, eef_pos, obstacle):
        a = np.array(action, dtype=float).copy()
        h, grad = self._barrier(eef_pos, obstacle)
        u = a[:3]
        lhs = float(grad @ u)
        rhs = -(self.alpha / self.action_scale) * h
        info = {'h': h, 'active': False, 'rhs': rhs, 'lhs': lhs}
        if lhs < rhs:
            gn2 = float(grad @ grad)
            if gn2 > 1e-12:
                u = u + ((rhs - lhs) / gn2) * grad
                a[:3] = u
                info['active'] = True
        return a, info


# class SafePolicyClient:
#     """Policy client with safety modes and dynamic parameter updates."""
    
#     def __init__(
#         self,
#         host: str,
#         port: int,
#         safety_mode: str = "none",
#         safety_margin: float = 0.01,
#         guidance_scale: float = 1.0,
#         cbf_iterations: int = 10,
#         cbf_step_size: float = 0.3
#     ):
#         self.client = _websocket_client_policy.WebsocketClientPolicy(host, port)
#         self.safety_mode = safety_mode
#         self.cbf_filter = PredictiveCBFFilter(
#             safety_margin=safety_margin,
#             max_iterations=cbf_iterations,
#             step_size=cbf_step_size,
#         )
#         self._guidance_scale = guidance_scale
#         self._safety_margin = safety_margin
    
#     @property
#     def guidance_scale(self) -> float:
#         return self._guidance_scale
    
#     @guidance_scale.setter
#     def guidance_scale(self, value: float):
#         self._guidance_scale = value
    
#     @property
#     def safety_margin(self) -> float:
#         return self._safety_margin
    
#     @safety_margin.setter
#     def safety_margin(self, value: float):
#         self._safety_margin = value
#         self.cbf_filter.update_safety_margin(value)
    
#     def infer(self, observation, eef_pos=None, obs_pos=None):
#         """Run inference with safety mode using current parameters."""
#         result = {'safety_info': {'mode': self.safety_mode, 'guidance_active': False}}
#         has_obstacle = eef_pos is not None and obs_pos is not None
        
#         if self.safety_mode == "none" or not has_obstacle:
#             result.update(self.client.infer(observation))
#         elif self.safety_mode == "flow":
#             result.update(self._infer_with_flow_guidance(observation, eef_pos, obs_pos))
#             result['safety_info']['guidance_active'] = True
#         elif self.safety_mode == "cbf":
#             result.update(self._infer_with_cbf(observation, eef_pos, obs_pos))
#         elif self.safety_mode == "flow+cbf":
#             result.update(self._infer_with_flow_guidance(observation, eef_pos, obs_pos))
#             actions = np.array(result['actions'])
#             safe_actions, cbf_info = self.cbf_filter.filter(actions, eef_pos, obs_pos)
#             result['actions'] = safe_actions
#             result['safety_info']['guidance_active'] = True
        
#         return result
    
#     def _infer_with_flow_guidance(self, observation, eef_pos, obs_pos):
#         observation = observation.copy()
#         observation['_safety_eef_pos'] = eef_pos.tolist()
#         observation['_safety_obs_pos'] = obs_pos.tolist()
#         observation['_safety_guidance_scale'] = self._guidance_scale
#         observation['_safety_margin'] = self._safety_margin
        
#         result = self.client.infer(observation)
#         result['safety_info'] = {'mode': 'flow', 'guidance_active': True}
#         return result
    
#     def _infer_with_cbf(self, observation, eef_pos, obs_pos):
#         result = self.client.infer(observation)
#         actions = np.array(result['actions'])
#         safe_actions, cbf_info = self.cbf_filter.filter(actions, eef_pos, obs_pos)
#         result['actions'] = safe_actions
#         result['safety_info'] = {'mode': 'cbf', 'cbf': cbf_info}
#         return result
class SafePolicyClient:
    """Policy client with safety modes and dynamic parameter updates."""
    
    def __init__(
        self,
        host: str,
        port: int,
        safety_mode: str = "none",
        safety_margin: float = 0.01,
        guidance_scale: float = 1.0,
        cbf_iterations: int = 10,
        cbf_step_size: float = 0.3,
        eef_scale: float = 1.0
    ):
        self.client = _websocket_client_policy.WebsocketClientPolicy(host, port)
        self.safety_mode = safety_mode
        self.cbf_filter = PredictiveCBFFilter(
            safety_margin=safety_margin,
            max_iterations=cbf_iterations,
            step_size=cbf_step_size,
        )
        # Single-action CBF (paper: arXiv 2512.11891), applied at execution time.
        self.single_cbf = SingleActionCBF(safety_margin=safety_margin, eef_scale=eef_scale)
        self._guidance_scale = guidance_scale
        self._safety_margin = safety_margin
    
    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale
    
    @guidance_scale.setter
    def guidance_scale(self, value: float):
        self._guidance_scale = value
    
    @property
    def safety_margin(self) -> float:
        return self._safety_margin
    
    @safety_margin.setter
    def safety_margin(self, value: float):
        self._safety_margin = value
        self.cbf_filter.update_safety_margin(value)
        self.single_cbf.update_safety_margin(value)
    
    def infer(self, observation, eef_pos=None, obstacles=None):
        """Run inference with safety mode using current parameters."""
        result = {'safety_info': {'mode': self.safety_mode, 'guidance_active': False}}
        has_obstacle = eef_pos is not None and obstacles is not None and len(obstacles) > 0
        
        if self.safety_mode == "none" or not has_obstacle:
            result.update(self.client.infer(observation))
        elif self.safety_mode == "flow":
            result.update(self._infer_with_flow_guidance(observation, eef_pos, obstacles))
            result['safety_info']['guidance_active'] = True
        elif self.safety_mode == "cbf":
            result.update(self._infer_with_cbf(observation, eef_pos, obstacles))
        elif self.safety_mode == "flow+cbf":
            result.update(self._infer_with_flow_guidance(observation, eef_pos, obstacles))
            actions = np.array(result['actions'])
            # CBF filter still uses closest surface point for post-hoc filtering
            obs_pos = self._closest_obstacle_point(eef_pos, obstacles)
            safe_actions, cbf_info = self.cbf_filter.filter(actions, eef_pos, obs_pos)
            result['actions'] = safe_actions
            result['safety_info']['guidance_active'] = True
        
        return result
    
    def _serialize_obstacles(self, obstacles: List[ObstacleGeometry]) -> list:
        """Serialize obstacles for websocket transport."""
        return [
            {
                'geom_type': int(obs.geom_type),
                'position': obs.position.tolist(),
                'rotation': obs.rotation.flatten().tolist(),
                'size': obs.size.tolist(),
            }
            for obs in obstacles
        ]
    
    def _infer_with_flow_guidance(self, observation, eef_pos, obstacles):
        observation = observation.copy()
        observation['_safety_eef_pos'] = eef_pos.tolist()
        observation['_safety_obstacles'] = self._serialize_obstacles(obstacles)
        observation['_safety_guidance_scale'] = self._guidance_scale
        observation['_safety_margin'] = self._safety_margin
        
        result = self.client.infer(observation)
        result['safety_info'] = {'mode': 'flow', 'guidance_active': True}
        return result
    
    def _infer_with_cbf(self, observation, eef_pos, obstacles):
        # Return the UNSAFE model actions; the single-action CBF (paper) is applied
        # per-step at execution time, not at the trajectory level.
        result = self.client.infer(observation)
        result['safety_info'] = {'mode': 'cbf'}
        return result
    
    def _closest_obstacle_point(self, eef_pos, obstacles):
        """Get closest obstacle surface point for legacy single-point APIs."""
        min_dist = float('inf')
        closest = eef_pos.copy()
        for obs in obstacles:
            dist = np.linalg.norm(eef_pos - obs.position)
            if dist < min_dist:
                min_dist = dist
                closest = obs.position.copy()
        return closest

# =============================================================================
# Main Evaluation
# =============================================================================

@dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    """Policy input image size (must stay 224 for the model)."""
    video_render_size: int = 512
    """Resolution of saved video frames (independent of policy input; up to env res 1024)."""
    replan_steps: int = 5

    task_suite_name: str = "safelibero_spatial"
    safety_level: Literal["I", "II"] = "I"
    task_index: Optional[int] = 3
    num_steps_wait: int = 30
    num_trials_per_task: int = 50

    safety_filter: Literal["none", "cbf", "flow", "flow+cbf"] = "none"
    safety_margin: float = 0.01
    guidance_scale: float = 1.0
    """Correction multiplier: 1.0 preserves solver output; other values are experimental."""
    obstacle_ellipsoid_scale: float = 1.0
    """Scale factor on the estimated obstacle ellipsoid semi-axes (<1 shrinks its volume)."""
    eef_ellipsoid_scale: float = 1.0
    """Scale factor on the end-effector ellipsoid radii used by the single-action CBF (<1 shrinks it)."""
    cbf_iterations: int = 10
    cbf_step_size: float = 0.3

    video_out_path: str = ""
    save_videos: bool = True
    save_only_failures: bool = False
    
    visualize: bool = True
    """Enable obstacle visualization in videos"""
    
    show_distance_bar: bool = True
    """Show distance bar in videos"""
    
    live_display: bool = False
    """Show live display of robot trials (requires display/X11)"""
    
    display_fps: int = 20
    """Target FPS for live display (controls delay between frames)"""
    
    display_scale: float = 2.0
    """Scale factor for live display window (1.0 = original size)"""
    
    pause_on_collision: bool = False
    """Pause live display when collision is detected"""
    
    pause_on_episode_end: bool = False
    """Pause live display at the end of each episode"""
    
    verbose: bool = False
    seed: int = 7

    remove_obstacles: bool = False
    """Delete obstacles from the scene (control / no-obstacle condition)"""

    only_episode: Optional[int] = None
    """If set, run only this single episode index (instead of 0..num_trials-1)."""
    save_trajectory: str = ""
    """If set, save the executed EEF world trajectory of the episode to this .npy path."""
    overlay_files: Tuple[str, ...] = ()
    """.npy EEF trajectories from prior conditions to overlay on the video."""
    overlay_colors: Tuple[str, ...] = ()
    """RGB colors (e.g. '255-0-0') matching overlay_files, one per file."""
    overlay_alpha: float = 0.55
    """Opacity of overlaid prior-condition trajectories (0..1)."""
    condition_tag: str = ""
    """If set, appended to saved video filenames to label the condition."""

    def __post_init__(self):
        if not self.video_out_path:
            self.video_out_path = f"data/safelibero/{self.safety_filter}/videos"


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _remove_obstacles_from_scene(env, obstacle_names):
    """Delete obstacles from the scene: hide their geoms and move their free
    joints far off-camera + underground so they cannot appear or collide."""
    model = env.sim.model
    data = env.sim.data
    n_removed = 0
    for name in obstacle_names:
        for gid, gname in enumerate(model.geom_names):
            if gname and gname.startswith(name):
                model.geom_rgba[gid, 3] = 0.0
        joint_name = f"{name}_joint0"
        try:
            addr = model.get_joint_qpos_addr(joint_name)
        except Exception:
            continue
        start = addr[0] if isinstance(addr, (tuple, list)) else addr
        data.qpos[start:start + 3] = [10.0, 10.0, -5.0]
        try:
            vaddr = model.get_joint_qvel_addr(joint_name)
            vstart = vaddr[0] if isinstance(vaddr, (tuple, list)) else vaddr
            data.qvel[vstart:vstart + 6] = 0.0
        except Exception:
            pass
        n_removed += 1
    env.sim.forward()
    print(f"[remove_obstacles] Deleted {n_removed} obstacle object(s) from scene")


def _get_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution
    )
    env.seed(seed)
    return env, task.language

def compute_obstacle_bounding_ellipsoid(env, tracker, trim_fraction=0.2):
    """
    Compute a single bounding ellipsoid for all obstacle geoms.
    Discards the furthest `trim_fraction` of surface sample points.
    Returns an ObstacleGeometry with geom_type=ELLIPSOID-approximated-as-BOX.
    """
    if not tracker.obstacle_geoms:
        return None
    
    # Collect surface sample points from all obstacle geoms
    all_points = []
    for obs_name, obs_data in tracker.obstacle_geoms.items():
        obs_pos, obs_rot = tracker.get_geom_pose(obs_data['geom_id'])
        offset = obs_data.get('pos_offset', None)
        if offset is not None:
            obs_pos = obs_pos + obs_rot @ offset
        
        geom_type = obs_data['geom_type']
        size = obs_data['size']
        
        # Skip geoms that are clearly far from workspace (x > 1m from origin)
        if np.linalg.norm(obs_pos[:2]) > 1.0:
            continue
        
        # Sample corners/surface points based on type
        if geom_type == 6:  # BOX
            half = size[:3]
            for sx in [-1, 0, 1]:
                for sy in [-1, 0, 1]:
                    for sz in [-1, 0, 1]:
                        local = np.array([sx * half[0], sy * half[1], sz * half[2]])
                        all_points.append(obs_rot @ local + obs_pos)
        elif geom_type == 7:  # MESH (approx as box)
            half = size[:3]
            for sx in [-1, 0, 1]:
                for sy in [-1, 0, 1]:
                    for sz in [-1, 0, 1]:
                        local = np.array([sx * half[0], sy * half[1], sz * half[2]])
                        all_points.append(obs_rot @ local + obs_pos)
        elif geom_type == 2:  # SPHERE
            r = size[0]
            for axis in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
                all_points.append(obs_pos + r * axis)
                all_points.append(obs_pos - r * axis)
        elif geom_type == 5:  # CYLINDER
            r, h = size[0], size[1]
            axis = obs_rot @ np.array([0, 0, 1])
            for sign in [-1, 1]:
                center = obs_pos + sign * h * axis
                perp1 = obs_rot @ np.array([1, 0, 0])
                perp2 = obs_rot @ np.array([0, 1, 0])
                all_points.append(center)
                all_points.append(center + r * perp1)
                all_points.append(center - r * perp1)
                all_points.append(center + r * perp2)
                all_points.append(center - r * perp2)
        else:
            all_points.append(obs_pos)
    
    if not all_points:
        return None
    
    points = np.array(all_points)
    center = np.mean(points, axis=0)
    
    # Trim outliers: discard furthest `trim_fraction` from center
    dists = np.linalg.norm(points - center, axis=1)
    cutoff = np.percentile(dists, (1 - trim_fraction) * 100)
    mask = dists <= cutoff
    trimmed = points[mask]
    
    if len(trimmed) < 4:
        trimmed = points  # fallback if too few points
    
    # Recompute center after trimming
    center = np.mean(trimmed, axis=0)
    
    # Compute axis-aligned bounding box of trimmed points as ellipsoid semi-axes
    deltas = trimmed - center
    half_extents = np.max(np.abs(deltas), axis=0)
    
    # Add small padding
    half_extents += 0.005
    
    return ObstacleGeometry(
        geom_type=4,  # treat as box for distance computation
        position=center,
        rotation=np.eye(3),  # axis-aligned
        size=half_extents,
    )
def compute_trajectory_from_actions(eef_pos, actions, dt=0.05):
    """Convert action chunk to trajectory positions."""
    traj = [eef_pos.copy()]
    pos = eef_pos.copy()
    for action in actions:
        pos = pos + action[:3] * dt
        traj.append(pos.copy())
    return np.array(traj)


def run_single_task(
    args: Args,
    client: SafePolicyClient,
    display_manager: LiveDisplayManager,
    task_suite_name: str,
    task_id: int,
    benchmark_dict: dict,
) -> Tuple[int, int, int, List[int], bool]:
    """
    Run evaluation for a single task.
    
    Returns:
        (episodes, successes, collisions, steps_list, should_switch_task)
    """
    task_suite = benchmark_dict[task_suite_name]()
    print(f"Running task suite: {task_suite_name}, task ID: {task_id}")

    if task_suite_name == "safelibero_object":
        z_limit = 0.45
        
    else:
        z_limit = 2.2
    max_steps = MAX_STEPS.get(task_suite_name, 500)
    
    try:
        task = task_suite.get_task(task_id, safety_level=args.safety_level)
        initial_states = task_suite.get_task_init_states(task_id, safety_level=args.safety_level)
    except TypeError:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
    
    env, task_description = _get_env(task, LIBERO_ENV_RESOLUTION, args.seed)


    sim = env.sim
    print(f"sim type: {type(sim)}")
    print(f"sim dir (render-related): {[x for x in dir(sim) if 'render' in x.lower() or 'context' in x.lower() or 'scn' in x.lower() or 'scene' in x.lower()]}")

    # Check if there's a render context already
    if hasattr(sim, '_render_context_offscreen'):
        ctx = sim._render_context_offscreen
        print(f"ctx type: {type(ctx)}")
        print(f"ctx dir: {[x for x in dir(ctx) if not x.startswith('__')]}")
        if hasattr(ctx, 'scn'):
            scn = ctx.scn
            print(f"scn type: {type(scn)}")
            print(f"ngeom: {scn.ngeom}, maxgeom: {scn.maxgeom}")

    # Also check what module sim comes from
    print(f"sim module: {type(sim).__module__}")
    if task_suite_name == "safelibero_goal" and task_id == 3:
        for i, name in enumerate(env.sim.model.geom_names):
            if name and 'plate' in name.lower():
                geom_type = env.sim.model.geom_type[i]
                geom_size = env.sim.model.geom_size[i]
                body_id = env.sim.model.geom_bodyid[i]
                pos = env.sim.data.geom_xpos[i]
                print(f"PLATE: {name}, type={geom_type}, size={geom_size}, pos={pos}")
    try:
        obstacle_names = []
        if hasattr(env, 'sim') and hasattr(env.sim, 'model'):
            obstacle_names = [str(n).replace('_joint0', '') for n in env.sim.model.joint_names if 'obstacle' in str(n)]

        task_episodes, task_successes, task_collisions = 0, 0, 0
        task_steps = []

        # Load prior-condition trajectories to overlay (world coords -> projected
        # each frame; valid because the agentview camera + world frame are shared).
        overlay_trajectories = []
        for i, fpath in enumerate(args.overlay_files):
            positions = np.load(fpath)
            if i < len(args.overlay_colors):
                color = tuple(int(c) for c in args.overlay_colors[i].split("-"))
            else:
                color = (255, 0, 0)
            overlay_trajectories.append({"positions": positions, "color": color})
            print(f"[overlay] loaded {len(positions)} pts from {fpath} color={color}")

        if args.only_episode is not None:
            episode_indices = [args.only_episode]
        else:
            episode_indices = list(range(min(args.num_trials_per_task, len(initial_states))))

        for episode_idx in episode_indices:
            if display_manager.is_quit_requested():
                break
            
            # Check for task change before starting episode
            if display_manager.has_pending_task_change():
                logging.info(f"Task change requested, stopping current task")
                return task_episodes, task_successes, task_collisions, task_steps, True
            
            logging.info(f"Task {task_id} Episode {episode_idx + 1}: {task_description}")
            
            # Episode restart loop
            while True:
                display_manager.reset_episode_flags()
                restart_episode = False
                force_failure = False
                
                env.reset()

                obs = env.set_init_state(initial_states[episode_idx])

                for step in range(args.num_steps_wait):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    for i, name in enumerate(env.sim.model.body_names):
                        if name and 'moka' in name.lower():
                            pos = env.sim.data.body_xpos[i].copy()
                            print(f">>> STEP {step+1} - {name}: z={pos[2]:.4f}")
                
                tracker = FullRobotGeometryTracker(env, verbose=args.verbose)
                bounding_obs = compute_obstacle_bounding_ellipsoid(env, tracker)
                if bounding_obs is not None and args.obstacle_ellipsoid_scale != 1.0:
                    bounding_obs.size = bounding_obs.size * args.obstacle_ellipsoid_scale
                    print(f"[obstacle_ellipsoid_scale] semi-axes x{args.obstacle_ellipsoid_scale} -> {bounding_obs.size}")
                obstacles = [bounding_obs] if bounding_obs is not None else []

                if obstacles:
                    logging.info(f"Obstacle bounding box: center={obstacles[0].position}, "
                 f"half_extents={obstacles[0].size}")
                print(obstacles)

                obstacle_name = ""
                initial_obs_pos = None
                for name in obstacle_names:
                    pos_key = f"{name}_pos"
                    if pos_key in obs:
                        p = obs[pos_key]
                        if p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
                            obstacle_name = name
                            initial_obs_pos = p.copy()
                            break

                if args.remove_obstacles:
                    _remove_obstacles_from_scene(env, obstacle_names)
                    tracker.obstacle_geoms = {}
                    obstacles = []
                    bounding_obs = None
                    obstacle_name = ""
                    initial_obs_pos = None

                action_plan = collections.deque()
                t = 0
                replay_images = []
                collide_flag = False
                collide_time = -1
                done = False
                action_chunk = None
                eef_history = []  # actual EEF world positions across the whole episode

                while t < max_steps + args.num_steps_wait:
                    # =====================================================
                    # UPDATE SAFETY PARAMETERS FROM DISPLAY MANAGER
                    # This allows real-time adjustment mid-episode
                    # =====================================================
                    if display_manager.enabled:
                        client.safety_margin = display_manager.safety_margin
                        client.guidance_scale = display_manager.guidance_scale
                    
                    # Check for task change during episode
                    if display_manager.has_pending_task_change():
                        logging.info(f"Task change requested during episode")
                        return task_episodes, task_successes, task_collisions, task_steps, True
                    
                    if display_manager.skip_episode or display_manager.is_quit_requested():
                        break
                    if display_manager.restart_requested:
                        restart_episode = True
                        display_manager.restart_requested = False
                        logging.info(f"Restarting episode {episode_idx + 1}")
                        break
                    if display_manager.skip_as_failure:
                        force_failure = True
                        display_manager.skip_as_failure = False
                        logging.info(f"Skipping episode {episode_idx + 1} as failure")
                        break
                        
                    try:
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        
                        img_resized = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )
                        eef_pos = obs["robot0_eef_pos"]
                        eef_history.append(np.array(eef_pos, dtype=float).copy())

                        robot_obs_dist = tracker.get_min_robot_obstacle_distance()

                        if robot_obs_dist is not None:
                            obs_pos = robot_obs_dist.obstacle_surface_point
                            
                            obs_info = GroundTruthObstacleInfo(
                                name=robot_obs_dist.obstacle_name,
                                geom_type=0,
                                position=robot_obs_dist.obstacle_surface_point,
                                size=np.zeros(3),
                                rotation=np.eye(3),
                                surface_point=robot_obs_dist.obstacle_surface_point,
                                distance=robot_obs_dist.distance,
                            )
                            
                            if args.verbose:
                                print(f"Closest: {robot_obs_dist.robot_geom_name} <-> {robot_obs_dist.obstacle_name}: {robot_obs_dist.distance:.4f}m")
                        else:
                            obs_info = None
                            obs_pos = None
                        
                        if not action_plan:
                            element = {
                                "observation/image": img_resized,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate((
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )),
                                "prompt": str(task_description),
                            }

                            result = client.infer(
                                element,
                                eef_pos=eef_pos,
                                obstacles=obstacles,
                            )
                            
                            action_chunk = result["actions"]
                            actions_to_use = action_chunk[:args.replan_steps]
                            action_plan.extend(actions_to_use)

                        # Use current safety margin from display for visualization
                        current_safety_margin = display_manager.safety_margin if display_manager.enabled else args.safety_margin
                        if args.visualize and action_chunk is not None:
                            predicted_trajectory = compute_trajectory_from_actions(
                                eef_pos=eef_pos,
                                actions=action_chunk,
                                dt=0.01
                            )
                            
                            visualizer = TrajectoryVisualizer(
                                env,
                                camera_name="agentview",
                                width=args.video_render_size,
                                height=args.video_render_size,
                                safety_margin=current_safety_margin
                            )
                            visualizer.set_predicted_trajectory(predicted_trajectory)
                            visualizer.set_history_trajectory(eef_history)
                            visualizer.set_overlay_trajectories(overlay_trajectories, args.overlay_alpha)
                            visualizer.set_obstacle_info(obs_info)
                            
                            vis_frame = visualizer.render_frame(
                                # obstacles=obstacles,
                                # eef_pos=eef_pos,
                                # eef_radii=np.array([0.06, 0.12, 0.11]),
                            )
                            
                            # if obs_info is not None:
                            #     vis_frame = draw_safety_overlay(
                            #         vis_frame, eef_pos, obs_info,
                            #         current_safety_margin, collide_flag
                            #     )
                            
                            replay_images.append(vis_frame)
                        else:
                            hires = env.sim.render(
                                width=args.video_render_size,
                                height=args.video_render_size,
                                camera_name="agentview",
                            )[::-1]
                            replay_images.append(np.ascontiguousarray(hires))

                        if args.live_display and replay_images:
                            display_frame = replay_images[-1]
                            episode_info = {
                                'task_id': task_id,
                                'episode': episode_idx + 1,
                                'step': t,
                                'max_steps': max_steps,
                                'collision': collide_flag,
                            }
                            
                            force_pause = args.pause_on_collision and collide_flag and collide_time == t
                            action = display_manager.show_frame(
                                display_frame,
                                episode_info=episode_info,
                                force_pause=force_pause,
                                wrist_frame=wrist_img,
                            )
                            
                            if action == 'quit':
                                break
                            elif action == 'skip':
                                break
                            elif action == 'restart':
                                restart_episode = True
                                logging.info(f"Restarting episode {episode_idx + 1}")
                                break
                            elif action == 'skip_fail':
                                force_failure = True
                                logging.info(f"Skipping episode {episode_idx + 1} as failure")
                                break
                            elif action == 'task_change':
                                logging.info(f"Task change requested via display")
                                return task_episodes, task_successes, task_collisions, task_steps, True

                        action = action_plan.popleft()
                        action = action.copy()

                        # Single-action CBF (paper arXiv:2512.11891): project the
                        # unsafe model action onto the CBF-safe half-space each step.
                        if args.safety_filter == "cbf" and obstacles:
                            action, cbf_info = client.single_cbf.filter(action, eef_pos, obstacles[0])
                            if cbf_info['active']:
                                print(f"[CBF] active: h={cbf_info['h']:.4f} corrected action")

                        action[3] = 0.0
                        action[4] = 0.0
                        action[5] = 0.0
                        
                        action_scale = 0.05
                        predicted_pos = eef_pos + action[:3] * action_scale
                        print(f"Predicted EEF Pos: {predicted_pos} \nAction: {action[:3]} \nEEF Pos: {eef_pos}\n")
                        
                        if eef_pos[2] > z_limit:
                            print(f"Z limit reached: {eef_pos[2]:.3f}")
                            if action[2] > 0.0:
                                action[2] = 0.0
                                                
                        if task_suite_name == "safelibero_object" and eef_pos[1] < -0.41:
                            print(f"Y limit reached: {eef_pos[1]:.3f}")
                            if action[1] < 0.0:
                                print("Restricting positive Y movement")
                                action[1] = 0.0
                        if task_suite_name == "safelibero_goal" and eef_pos[1] > 0.41:
                            print(f"Y limit reached: {eef_pos[1]:.3f}")
                            if action[1] > 0.0:
                                print("Restricting negative Y movement")
                                action[1] = 0.0

                        if task_suite_name == "safelibero_spatial":
                            if task_id == 2 and t <= 10:
                                print("Activated")
                                action[0] = -action[0]
                        
                        obs, reward, done, info = env.step(
                            action.tolist() if hasattr(action, 'tolist') else action
                        )
                        t += 1
                        
                        if not collide_flag and initial_obs_pos is not None:
                            curr_pos = obs[f"{obstacle_name}_pos"]
                            displacement = np.sum(np.abs(curr_pos - initial_obs_pos))
                            
                            if displacement > 0.001:
                                if args.verbose:
                                    logging.info(f"Collision at step {t}")
                                collide_flag, collide_time = True, t

                        if done:
                            task_successes += 1
                            break

                    except Exception as e:
                        logging.error(f"Exception: {e}")
                        import traceback
                        traceback.print_exc()
                        break

                if restart_episode:
                    continue
                
                if force_failure:
                    done = False
                
                # Normal exit from episode
                break

            # End of episode pause
            if args.live_display and args.pause_on_episode_end and replay_images:
                episode_info = {
                    'task_id': task_id,
                    'episode': episode_idx + 1,
                    'step': t,
                    'max_steps': max_steps,
                    'collision': collide_flag,
                }
                action = display_manager.show_frame(
                    replay_images[-1],
                    episode_info=episode_info,
                    force_pause=True,
                    wrist_frame=None,
                )
                if action == 'task_change':
                    return task_episodes, task_successes, task_collisions, task_steps, True

            if collide_flag:
                task_collisions += 1

            task_episodes += 1
            task_steps.append(t)

            if args.verbose or (not done) or collide_flag:
                logging.info(
                    f"Episode {episode_idx + 1}: Success={done}, "
                    f"Collision={collide_flag}, Steps={t}"
                )

            if args.save_videos:
                should_save = not args.save_only_failures or (not done or collide_flag)
                if should_save and replay_images:
                    suffix = "success" if done else "failure"
                    coll_str = "_collision" if collide_flag else ""
                    task_seg = task_description.replace(" ", "_")[:40]
                    tag_str = f"_{args.condition_tag}" if args.condition_tag else ""
                    video_path = pathlib.Path(args.video_out_path) / f"{task_seg}_{episode_idx}_{suffix}{coll_str}{tag_str}.mp4"
                    imageio.mimwrite(str(video_path), [np.asarray(x) for x in replay_images], fps=10)

            if args.save_trajectory and eef_history:
                pathlib.Path(args.save_trajectory).parent.mkdir(parents=True, exist_ok=True)
                np.save(args.save_trajectory, np.asarray(eef_history))
                print(f"[save_trajectory] wrote {len(eef_history)} pts to {args.save_trajectory}")

        if task_episodes > 0:
            logging.info(
                f"Task {task_id}: Success {task_successes}/{task_episodes} "
                f"({100*task_successes/task_episodes:.1f}%), "
                f"Collisions {task_collisions}/{task_episodes} "
                f"({100*task_collisions/task_episodes:.1f}%)"
            )
        
        return task_episodes, task_successes, task_collisions, task_steps, False
        
    finally:
        env.close()


def eval_safelibero(args: Args) -> None:
    np.random.seed(args.seed)

    # Current task state
    current_task_suite = args.task_suite_name
    current_task_id = args.task_index - 1 if args.task_index is not None else 0
    
    benchmark_dict = benchmark.get_benchmark_dict()
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"Video output path: {pathlib.Path(args.video_out_path).resolve()}")
    logging.info(f"Visualization: {'ENABLED' if args.visualize else 'DISABLED'}")
    logging.info(f"Live display: {'ENABLED' if args.live_display else 'DISABLED'}")

    client = SafePolicyClient(
        host=args.host,
        port=args.port,
        safety_mode=args.safety_filter,
        safety_margin=args.safety_margin,
        guidance_scale=args.guidance_scale,
        cbf_iterations=args.cbf_iterations,
        cbf_step_size=args.cbf_step_size,
        eef_scale=args.eef_ellipsoid_scale,
    )

    # Initialize live display manager with dropdowns and sliders
    display_manager = LiveDisplayManager(
        window_name=f"SafeLIBERO Live",
        scale=args.display_scale,
        target_fps=args.display_fps,
        enabled=args.live_display,
        task_suites=TASK_SUITES,
        initial_task_suite=current_task_suite,
        initial_task_id=current_task_id,
        initial_safety_margin=args.safety_margin,
        initial_guidance_scale=args.guidance_scale,
    )

    logging.info(f"Initial task suite: {current_task_suite}, Task ID: {current_task_id}")
    logging.info(f"Safety mode: {args.safety_filter}")
    logging.info(f"Initial safety margin: {args.safety_margin}, guidance scale: {args.guidance_scale}")

    total_episodes, total_successes, total_collisions = 0, 0, 0
    total_steps = []

    try:
        while not display_manager.is_quit_requested():
            # Update display to show current task
            display_manager.set_current_task(current_task_suite, current_task_id)
            
            # Update task ID dropdown options for current suite
            if display_manager.enabled:
                try:
                    task_suite = benchmark_dict[current_task_suite]()
                    num_tasks = len(task_suite.tasks) if hasattr(task_suite, 'tasks') else 10
                except:
                    num_tasks = 10
                display_manager.update_task_id_options(list(range(1, num_tasks + 1)))
            
            logging.info(f"=" * 50)
            logging.info(f"Running: {current_task_suite} Task {current_task_id}")
            logging.info(f"=" * 50)
            
            # Run the current task
            episodes, successes, collisions, steps, should_switch = run_single_task(
                args=args,
                client=client,
                display_manager=display_manager,
                task_suite_name=current_task_suite,
                task_id=current_task_id,
                benchmark_dict=benchmark_dict,
            )
            
            # Accumulate stats
            total_episodes += episodes
            total_successes += successes
            total_collisions += collisions
            total_steps.extend(steps)
            
            # Handle task switch
            if should_switch:
                new_suite, new_task_id = display_manager.get_and_clear_pending_task_change()
                
                if new_suite is not None:
                    current_task_suite = new_suite
                    logging.info(f"Switched to task suite: {current_task_suite}")
                
                if new_task_id is not None:
                    current_task_id = new_task_id
                    logging.info(f"Switched to task ID: {current_task_id}")
                
                continue  # Go back to top of loop with new task
            
            # If we didn't switch, check if we should wait for user input or exit
            if args.live_display:
                logging.info("Task complete. Select a new task from dropdown or press 'q' to quit.")
                
                # Wait for user to either select a new task or quit
                while not display_manager.is_quit_requested():
                    if display_manager.has_pending_task_change():
                        new_suite, new_task_id = display_manager.get_and_clear_pending_task_change()
                        
                        if new_suite is not None:
                            current_task_suite = new_suite
                        if new_task_id is not None:
                            current_task_id = new_task_id
                        
                        break  # Exit wait loop, continue main loop
                    
                    # Poll for events
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q') or key == 27:
                        display_manager.quit_requested = True
                        break
            else:
                # Non-interactive mode - just exit
                break

    finally:
        display_manager.close()

    # Print final summary
    if total_episodes > 0:
        tsr = 100 * total_successes / total_episodes
        car = 100 * (total_episodes - total_collisions) / total_episodes
        ets = np.mean(total_steps) if total_steps else 0
        
        logging.info("=" * 60)
        logging.info(f"FINAL RESULTS")
        logging.info(f"Safety Mode: {args.safety_filter}")
        logging.info("-" * 60)
        logging.info(f"Total Episodes: {total_episodes}")
        logging.info(f"Task Success Rate (TSR):       {tsr:.1f}%")
        logging.info(f"Collision Avoidance Rate (CAR): {car:.1f}%")
        logging.info(f"Execution Time Steps (ETS):    {ets:.1f}")
        logging.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    args = tyro.cli(Args)
    eval_safelibero(args)