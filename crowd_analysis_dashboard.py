import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
from datetime import datetime, timedelta
import threading
from PIL import Image, ImageTk
import pandas as pd

class ZoneConfigurationTool:
    """Interactive tool for configuring crowd analysis zones"""
    
    def __init__(self, master):
        self.master = master
        self.master.title("Crowd Analysis Zone Configuration")
        self.master.geometry("1200x800")
        
        # Variables
        self.current_image = None
        self.zones = []
        self.current_zone_points = []
        self.drawing_mode = False
        self.zone_counter = 1
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Zone Configuration", width=300)
        control_frame.pack(side='left', fill='y', padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Load image button
        ttk.Button(control_frame, text="Load Background Image", 
                  command=self.load_image).pack(pady=5, fill='x')
        
        # Zone creation
        zone_frame = ttk.LabelFrame(control_frame, text="Create Zone")
        zone_frame.pack(fill='x', pady=5)
        
        ttk.Label(zone_frame, text="Zone Name:").pack(anchor='w')
        self.zone_name_var = tk.StringVar(value=f"Zone_{self.zone_counter}")
        ttk.Entry(zone_frame, textvariable=self.zone_name_var).pack(fill='x', pady=2)
        
        ttk.Label(zone_frame, text="Alert Threshold:").pack(anchor='w')
        self.threshold_var = tk.StringVar(value="15")
        ttk.Entry(zone_frame, textvariable=self.threshold_var).pack(fill='x', pady=2)
        
        ttk.Label(zone_frame, text="Zone Type:").pack(anchor='w')
        self.zone_type_var = tk.StringVar(value="Rectangle")
        zone_type_combo = ttk.Combobox(zone_frame, textvariable=self.zone_type_var,
                                      values=["Rectangle", "Polygon"], state="readonly")
        zone_type_combo.pack(fill='x', pady=2)
        
        ttk.Button(zone_frame, text="Start Drawing Zone", 
                  command=self.start_zone_drawing).pack(pady=5, fill='x')
        
        ttk.Button(zone_frame, text="Finish Zone", 
                  command=self.finish_zone).pack(pady=2, fill='x')
        
        ttk.Button(zone_frame, text="Clear Current Zone", 
                  command=self.clear_current_zone).pack(pady=2, fill='x')
        
        # Zone list
        list_frame = ttk.LabelFrame(control_frame, text="Configured Zones")
        list_frame.pack(fill='both', expand=True, pady=5)
        
        # Treeview for zones
        self.zone_tree = ttk.Treeview(list_frame, columns=('Type', 'Threshold'), show='tree headings')
        self.zone_tree.heading('#0', text='Zone Name')
        self.zone_tree.heading('Type', text='Type')
        self.zone_tree.heading('Threshold', text='Threshold')
        self.zone_tree.column('#0', width=120)
        self.zone_tree.column('Type', width=80)
        self.zone_tree.column('Threshold', width=80)
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.zone_tree.yview)
        self.zone_tree.configure(yscrollcommand=scrollbar.set)
        
        self.zone_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Buttons frame
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill='x', pady=5)
        
        ttk.Button(btn_frame, text="Delete Selected", 
                  command=self.delete_selected_zone).pack(side='left', padx=2)
        
        ttk.Button(btn_frame, text="Save Config", 
                  command=self.save_configuration).pack(side='right', padx=2)
        
        ttk.Button(btn_frame, text="Load Config", 
                  command=self.load_configuration).pack(side='right', padx=2)
        
        # Right panel - Image canvas
        canvas_frame = ttk.LabelFrame(main_frame, text="Zone Preview")
        canvas_frame.pack(side='right', fill='both', expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(canvas_frame, bg='white', width=800, height=600)
        self.canvas.pack(fill='both', expand=True, padx=5, pady=5)
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        self.canvas.bind('<Motion>', self.on_canvas_motion)
        
        # Status bar
        self.status_var = tk.StringVar(value="Load an image to start configuring zones")
        status_bar = ttk.Label(self.master, textvariable=self.status_var, relief='sunken')
        status_bar.pack(side='bottom', fill='x')
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Background Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image()
            self.status_var.set(f"Loaded image: {file_path}")
    
    def display_image(self):
        if self.current_image is None:
            return
        
        # Resize image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.master.after(100, self.display_image)
            return
        
        h, w = self.current_image.shape[:2]
        
        # Calculate scaling factor
        scale_w = canvas_width / w
        scale_h = canvas_height / h
        scale = min(scale_w, scale_h) * 0.9  # Leave some margin
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Store scaling info
        self.scale_factor = scale
        self.display_size = (new_w, new_h)
        self.original_size = (w, h)
        
        # Resize and display
        resized_image = cv2.resize(self.current_image, (new_w, new_h))
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, anchor='center', image=self.photo)
        
        # Draw existing zones
        self.draw_zones()
    
    # Missing continuation of start_zone_drawing method and other methods

    def start_zone_drawing(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.drawing_mode = True
        self.current_zone_points = []
        self.status_var.set(f"Drawing {self.zone_type_var.get().lower()}. Click to add points.")
    
    def on_canvas_click(self, event):
        if not self.drawing_mode or self.current_image is None:
            return
        
        # Convert canvas coordinates to image coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Calculate image position on canvas
        img_x = (canvas_width - self.display_size[0]) // 2
        img_y = (canvas_height - self.display_size[1]) // 2
        
        # Check if click is within image bounds
        click_x = event.x - img_x
        click_y = event.y - img_y
        
        if (0 <= click_x <= self.display_size[0] and 
            0 <= click_y <= self.display_size[1]):
            
            # Convert to original image coordinates
            orig_x = int(click_x / self.scale_factor)
            orig_y = int(click_y / self.scale_factor)
            
            self.current_zone_points.append((orig_x, orig_y))
            
            # For rectangle, only need 2 points
            if self.zone_type_var.get() == "Rectangle" and len(self.current_zone_points) == 2:
                self.finish_zone()
                return
            
            # Update display
            self.draw_zones()
            self.draw_current_zone()
            
            self.status_var.set(f"Points: {len(self.current_zone_points)}. " + 
                              ("Right-click to finish polygon." if self.zone_type_var.get() == "Polygon" else ""))
        
        # Right-click to finish polygon
        if event.num == 3 and self.zone_type_var.get() == "Polygon" and len(self.current_zone_points) >= 3:
            self.finish_zone()
    
    def on_canvas_motion(self, event):
        if not self.drawing_mode or not self.current_zone_points:
            return
        
        # For rectangle, show preview of second point
        if (self.zone_type_var.get() == "Rectangle" and 
            len(self.current_zone_points) == 1):
            self.draw_zones()
            self.draw_current_zone()
            self.draw_preview_point(event)
    
    def draw_preview_point(self, event):
        # Convert canvas coordinates to image coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        img_x = (canvas_width - self.display_size[0]) // 2
        img_y = (canvas_height - self.display_size[1]) // 2
        
        click_x = event.x - img_x
        click_y = event.y - img_y
        
        if (0 <= click_x <= self.display_size[0] and 
            0 <= click_y <= self.display_size[1]):
            
            # Draw preview rectangle
            start_point = self.current_zone_points[0]
            start_x = int(start_point[0] * self.scale_factor) + img_x
            start_y = int(start_point[1] * self.scale_factor) + img_y
            
            self.canvas.create_rectangle(start_x, start_y, event.x, event.y,
                                       outline='yellow', width=2, tags='preview')
    
    def draw_zones(self):
        # Clear existing zone drawings
        self.canvas.delete("zone")
        
        if not hasattr(self, 'display_size'):
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_x = (canvas_width - self.display_size[0]) // 2
        img_y = (canvas_height - self.display_size[1]) // 2
        
        for i, zone in enumerate(self.zones):
            color = ['red', 'blue', 'green', 'purple', 'orange'][i % 5]
            coords = zone['coordinates']
            
            if len(coords) >= 3:  # Polygon
                canvas_coords = []
                for x, y in coords:
                    canvas_x = int(x * self.scale_factor) + img_x
                    canvas_y = int(y * self.scale_factor) + img_y
                    canvas_coords.extend([canvas_x, canvas_y])
                
                self.canvas.create_polygon(canvas_coords, outline=color, 
                                         fill='', width=2, tags='zone')
            else:  # Rectangle
                x1, y1 = coords[0]
                x2, y2 = coords[1]
                canvas_x1 = int(x1 * self.scale_factor) + img_x
                canvas_y1 = int(y1 * self.scale_factor) + img_y
                canvas_x2 = int(x2 * self.scale_factor) + img_x
                canvas_y2 = int(y2 * self.scale_factor) + img_y
                
                self.canvas.create_rectangle(canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                                           outline=color, width=2, tags='zone')
            
            # Add zone label
            if coords:
                label_x = int(coords[0][0] * self.scale_factor) + img_x
                label_y = int(coords[0][1] * self.scale_factor) + img_y - 10
                self.canvas.create_text(label_x, label_y, text=zone['name'], 
                                      fill=color, font=('Arial', 10, 'bold'), tags='zone')
    
    def draw_current_zone(self):
        # Clear preview drawings
        self.canvas.delete("current")
        self.canvas.delete("preview")
        
        if not self.current_zone_points or not hasattr(self, 'display_size'):
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_x = (canvas_width - self.display_size[0]) // 2
        img_y = (canvas_height - self.display_size[1]) // 2
        
        # Draw points
        for i, (x, y) in enumerate(self.current_zone_points):
            canvas_x = int(x * self.scale_factor) + img_x
            canvas_y = int(y * self.scale_factor) + img_y
            
            self.canvas.create_oval(canvas_x-3, canvas_y-3, canvas_x+3, canvas_y+3,
                                  fill='yellow', outline='red', width=2, tags='current')
            
            # Add point number
            self.canvas.create_text(canvas_x+10, canvas_y-10, text=str(i+1),
                                  fill='yellow', font=('Arial', 8, 'bold'), tags='current')
        
        # Draw lines connecting points
        if len(self.current_zone_points) > 1:
            for i in range(len(self.current_zone_points)):
                x1, y1 = self.current_zone_points[i]
                x2, y2 = self.current_zone_points[(i+1) % len(self.current_zone_points)]
                
                canvas_x1 = int(x1 * self.scale_factor) + img_x
                canvas_y1 = int(y1 * self.scale_factor) + img_y
                canvas_x2 = int(x2 * self.scale_factor) + img_x
                canvas_y2 = int(y2 * self.scale_factor) + img_y
                
                # Only draw closing line for polygons with 3+ points
                if i == len(self.current_zone_points) - 1 and len(self.current_zone_points) < 3:
                    continue
                
                self.canvas.create_line(canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                                      fill='yellow', width=2, tags='current')
    
    def finish_zone(self):
        if not self.current_zone_points:
            messagebox.showwarning("Warning", "No points selected")
            return
        
        zone_type = self.zone_type_var.get()
        
        # Validate points
        if zone_type == "Rectangle" and len(self.current_zone_points) != 2:
            messagebox.showwarning("Warning", "Rectangle requires exactly 2 points")
            return
        
        if zone_type == "Polygon" and len(self.current_zone_points) < 3:
            messagebox.showwarning("Warning", "Polygon requires at least 3 points")
            return
        
        # Create zone
        zone_data = {
            'name': self.zone_name_var.get(),
            'type': zone_type,
            'coordinates': self.current_zone_points.copy(),
            'threshold': float(self.threshold_var.get())
        }
        
        self.zones.append(zone_data)
        
        # Add to tree view
        self.zone_tree.insert('', 'end', text=zone_data['name'], 
                             values=(zone_data['type'], zone_data['threshold']))
        
        # Reset for next zone
        self.drawing_mode = False
        self.current_zone_points = []
        self.zone_counter += 1
        self.zone_name_var.set(f"Zone_{self.zone_counter}")
        
        # Update display
        self.draw_zones()
        self.canvas.delete("current")
        self.canvas.delete("preview")
        
        self.status_var.set(f"Zone '{zone_data['name']}' created successfully")
    
    def clear_current_zone(self):
        self.current_zone_points = []
        self.drawing_mode = False
        self.canvas.delete("current")
        self.canvas.delete("preview")
        self.status_var.set("Current zone cleared")
    
    def delete_selected_zone(self):
        selected_items = self.zone_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select a zone to delete")
            return
        
        for item in selected_items:
            zone_name = self.zone_tree.item(item, 'text')
            
            # Remove from zones list
            self.zones = [z for z in self.zones if z['name'] != zone_name]
            
            # Remove from tree view
            self.zone_tree.delete(item)
        
        # Update display
        self.draw_zones()
        self.status_var.set("Selected zone(s) deleted")
    
    def save_configuration(self):
        if not self.zones:
            messagebox.showwarning("Warning", "No zones to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Zone Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            config_data = {
                'zones': self.zones,
                'created_at': datetime.now().isoformat(),
                'image_size': self.original_size if hasattr(self, 'original_size') else None
            }
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Configuration saved to {file_path}")
                self.status_var.set(f"Configuration saved: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def load_configuration(self):
        file_path = filedialog.askopenfilename(
            title="Load Zone Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
                
                # Clear existing zones
                self.zones = []
                for item in self.zone_tree.get_children():
                    self.zone_tree.delete(item)
                
                # Load zones
                if 'zones' in config_data:
                    self.zones = config_data['zones']
                    
                    # Populate tree view
                    for zone in self.zones:
                        self.zone_tree.insert('', 'end', text=zone['name'],
                                            values=(zone['type'], zone['threshold']))
                    
                    # Update zone counter
                    if self.zones:
                        max_zone_num = max([int(z['name'].split('_')[-1]) 
                                          for z in self.zones 
                                          if z['name'].startswith('Zone_')], default=0)
                        self.zone_counter = max_zone_num + 1
                        self.zone_name_var.set(f"Zone_{self.zone_counter}")
                
                # Update display
                self.draw_zones()
                
                messagebox.showinfo("Success", f"Configuration loaded from {file_path}")
                self.status_var.set(f"Configuration loaded: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")

# Main application runner
class CrowdAnalysisDashboard:
    """Main dashboard application combining zone configuration and heatmap analysis"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Crowd Analysis Dashboard")
        self.root.geometry("1400x900")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Zone Configuration Tab
        self.zone_config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.zone_config_frame, text="Zone Configuration")
        self.zone_config_tool = ZoneConfigurationTool(self.zone_config_frame)
        
        # Real-time Analysis Tab
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Real-time Analysis")
        self.setup_analysis_tab()
        
        # Historical Data Tab
        self.history_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.history_frame, text="Historical Data")
        self.setup_history_tab()
    
    def setup_analysis_tab(self):
        """Setup the real-time analysis tab"""
        # Control panel
        control_panel = ttk.LabelFrame(self.analysis_frame, text="Analysis Controls")
        control_panel.pack(fill='x', padx=10, pady=5)
        
        # Model selection
        ttk.Label(control_panel, text="Model Path:").pack(side='left', padx=5)
        self.model_path_var = tk.StringVar()
        ttk.Entry(control_panel, textvariable=self.model_path_var, width=50).pack(side='left', padx=5)
        ttk.Button(control_panel, text="Browse", command=self.browse_model).pack(side='left', padx=5)
        
        # Zone config selection
        ttk.Label(control_panel, text="Zone Config:").pack(side='left', padx=5)
        self.zone_config_path_var = tk.StringVar()
        ttk.Entry(control_panel, textvariable=self.zone_config_path_var, width=30).pack(side='left', padx=5)
        ttk.Button(control_panel, text="Browse", command=self.browse_zone_config).pack(side='left', padx=5)
        
        # Start/Stop buttons
        ttk.Button(control_panel, text="Start Analysis", command=self.start_analysis).pack(side='right', padx=5)
        ttk.Button(control_panel, text="Stop Analysis", command=self.stop_analysis).pack(side='right', padx=5)
        
        # Status display
        self.analysis_status_var = tk.StringVar(value="Ready to start analysis")
        ttk.Label(self.analysis_frame, textvariable=self.analysis_status_var).pack(pady=5)
        
        # Placeholder for analysis results
        self.analysis_text = tk.Text(self.analysis_frame, height=20, width=80)
        self.analysis_text.pack(fill='both', expand=True, padx=10, pady=5)
    
    def setup_history_tab(self):
        """Setup the historical data tab"""
        # Placeholder for historical analysis
        ttk.Label(self.history_frame, text="Historical Data Analysis", 
                 font=('Arial', 16, 'bold')).pack(pady=20)
        
        # Data loading controls
        data_frame = ttk.LabelFrame(self.history_frame, text="Load Historical Data")
        data_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(data_frame, text="Load CSV Data", command=self.load_csv_data).pack(side='left', padx=5)
        ttk.Button(data_frame, text="Generate Report", command=self.generate_report).pack(side='left', padx=5)
        
        # Results area
        self.history_text = tk.Text(self.history_frame, height=25, width=100)
        self.history_text.pack(fill='both', expand=True, padx=10, pady=5)
    
    def browse_model(self):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
    
    def browse_zone_config(self):
        file_path = filedialog.askopenfilename(
            title="Select Zone Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.zone_config_path_var.set(file_path)
    
    def start_analysis(self):
        model_path = self.model_path_var.get()
        zone_config_path = self.zone_config_path_var.get()
        
        if not model_path:
            messagebox.showwarning("Warning", "Please select a model file")
            return
        
        # Load zone configuration if provided
        zones_config = None
        if zone_config_path:
            try:
                with open(zone_config_path, 'r') as f:
                    config_data = json.load(f)
                    zones_config = config_data.get('zones', [])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load zone configuration: {e}")
                return
        
        self.analysis_status_var.set("Starting analysis...")
        self.analysis_text.insert(tk.END, f"Starting crowd analysis with model: {model_path}\n")
        
        # Run analysis in separate thread
        def run_analysis():
            try:
                # This would integrate with the RealTimeCrowdHeatmap class
                heatmap_system = RealTimeCrowdHeatmap(model_path, zones_config)
                # For demo purposes, just show that it would start
                self.analysis_text.insert(tk.END, "Analysis system initialized successfully\n")
                self.analysis_text.insert(tk.END, "Note: Actual video processing would run in separate window\n")
                self.analysis_status_var.set("Analysis running...")
            except Exception as e:
                self.analysis_text.insert(tk.END, f"Error: {e}\n")
                self.analysis_status_var.set("Analysis failed")
        
        threading.Thread(target=run_analysis, daemon=True).start()
    
    def stop_analysis(self):
        self.analysis_status_var.set("Analysis stopped")
        self.analysis_text.insert(tk.END, "Analysis stopped by user\n")
    
    def load_csv_data(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                df = pd.read_csv(file_path)
                self.history_text.insert(tk.END, f"Loaded data from: {file_path}\n")
                self.history_text.insert(tk.END, f"Data shape: {df.shape}\n")
                self.history_text.insert(tk.END, f"Columns: {list(df.columns)}\n\n")
                self.history_text.insert(tk.END, "First 5 rows:\n")
                self.history_text.insert(tk.END, df.head().to_string())
                self.history_text.insert(tk.END, "\n\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {e}")
    
    def generate_report(self):
        self.history_text.insert(tk.END, "Generating analysis report...\n")
        self.history_text.insert(tk.END, "Report generation feature would be implemented here\n")
    
    def run(self):
        self.root.mainloop()

# Entry point
if __name__ == "__main__":
    app = CrowdAnalysisDashboard()
    app.run()