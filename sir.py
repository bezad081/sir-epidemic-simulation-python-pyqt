

import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from PyQt5.QtGui import QFont, QDesktopServices
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QComboBox



class SIRModel(QWidget):
    def __init__(self):
        super().__init__()
        
        # Set font
        font = QFont()
        font.setPointSize(12)
        self.setFont(font)
        
        # Set stylesheet
        self.setStyleSheet("""
            QLabel {
                color: #444444;
            }
            QLineEdit {
                border: 2px solid #aaaaaa;
                border-radius: 5px;
                padding: 5px;
                font-size: 12pt;
            }
            QPushButton {
                background-color: #4caf50;
                border: none;
                border-radius: 5px;
                color: white;
                padding: 10px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #3e8e41;
            }
            QComboBox {
                border: 2px solid #aaaaaa;
                border-radius: 5px;
                padding: 5px;
                font-size: 12pt;
            }
            """)
        
        # Create UI elements
        self.title = QLabel("SIR Model Simulatoion")

                # Add hyperlink label
        self.link_label = QLabel('<a href="https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model">Learn more about SIR model equations</a>')
        self.link_label.setOpenExternalLinks(True)
        
        self.t0_label = QLabel("Initial time (days):")
        self.t0_input = QLineEdit()
        
        self.tf_label = QLabel("Final time (days):")
        self.tf_input = QLineEdit()
        
        self.dt_label = QLabel("Step size ( < 0.1) :")
        self.dt_input = QLineEdit()
        
        self.beta_label = QLabel("Infection rate (bete =< 1)  :")
        self.beta_input = QLineEdit()
        
        self.gamma_label = QLabel("Recovery rate (gamma < 0.1):")
        self.gamma_input = QLineEdit()
        
        self.s0_label = QLabel("Initial susceptible population:")
        self.s0_input = QLineEdit()
        
        self.i0_label = QLabel("Initial infectious population:")
        self.i0_input = QLineEdit()
        
        self.method_label = QLabel("Integration method:")
        self.method_input = QComboBox()
        self.method_input.addItem("Euler")
        self.method_input.addItem("Runge-Kutta")
        
        self.run_button = QPushButton("Run")
        self.output_label = QLabel("")

        # Connect button to function
        self.run_button.clicked.connect(self.run)
        
        # Set layout
        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.link_label)
        layout.addWidget(self.t0_label)
        layout.addWidget(self.t0_input)
        layout.addWidget(self.tf_label)
        layout.addWidget(self.tf_input)
        layout.addWidget(self.dt_label)
        layout.addWidget(self.dt_input)
        layout.addWidget(self.beta_label)
        layout.addWidget(self.beta_input)
        layout.addWidget(self.gamma_label)
        layout.addWidget(self.gamma_input)
        layout.addWidget(self.s0_label)
        layout.addWidget(self.s0_input)
        layout.addWidget(self.i0_label)
        layout.addWidget(self.i0_input)
        layout.addWidget(self.method_label)
        layout.addWidget(self.method_input)
        layout.addWidget(self.run_button)
        layout.addWidget(self.output_label)
  
        
        self.setLayout(layout)
    
    def sir_eq(self, x, t, beta, gamma, N):
        s, e, i, h, r, d = x
        dsdt = -beta * s * i / N
        dedt = beta * s * i / N - e * 0.1
        dhdt = e * 0.03 - h * 0.1
        didt = e * 0.7 - i * 0.1
        drdt = i * 0.1 + h * 0.07
        dddt = i * 0.03
        return np.array([dsdt, dedt, didt, dhdt, drdt, dddt])
    
    def euler_step(self, x, t, f, dt, *args):
        x_new = x + dt * f(x, t, *args)
        return x_new
    
    def rk4_step(self, x, t, f, dt, *args):
        k1 = dt * f(x, t, *args)
        k2 = dt * f(x + 0.5 * k1, t + 0.5 * dt, *args)
        k3 = dt * f(x + 0.5 * k2, t + 0.5 * dt, *args)
        k4 = dt * f(x + k3, t + dt, *args)
        x_new = x + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x_new
    
    def run(self):
        # Get inputs from UI
        t0 = float(self.t0_input.text())
        tf = float(self.tf_input.text())
        dt = float(self.dt_input.text())
        beta = float(self.beta_input.text())
        gamma = float(self.gamma_input.text())
        s0 = float(self.s0_input.text())
        i0 = float(self.i0_input.text())
        N = s0 + i0
        
        # Set initial conditions
        x0 = np.array([s0, 0.0, i0, 0.0, 0.0, 0.0]) # susceptible, exposed, infectious, hospitalized, recovered, and dead populations
        t = np.arange(t0, tf + dt, dt)
        N = len(t)
        x = np.empty((N, 6))
        x[0] = x0
        
        # Integrate the SIR equations using the selected method

        
        if self.method_input.currentText() == "Euler":
            print("euler")
            for n in range(N - 1):
                x[n + 1] = self.euler_step(x[n], t[n], self.sir_eq, dt, beta, gamma,N)
        elif self.method_input.currentText() == "Runge-Kutta":
            print('not euler')
            for n in range(N - 1):
                x[n + 1] = self.rk4_step(x[n], t[n], self.sir_eq, dt, beta, gamma,N)
        
        # Create the plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=x[:,0], mode='lines', name='Susceptible'))
        fig.add_trace(go.Scatter(x=t, y=x[:,1],mode='lines', name='Exposed'))
        fig.add_trace(go.Scatter(x=t, y=x[:,2], mode='lines', name='Infectious'))
        fig.add_trace(go.Scatter(x=t, y=x[:,3],mode='lines', name='Hospitalized'))
        fig.add_trace(go.Scatter(x=t, y=x[:,4], mode='lines', name='Recovered'))
        fig.add_trace(go.Scatter(x=t, y=x[:,5], mode='lines', name='Dead'))
        fig.update_layout(title='SIR Epidemic Model', xaxis_title='Time (days)', yaxis_title='Population')

            # Show the plot
        pyo.iplot(fig)
        self.output_label.setText("Simulation complete.")
        
if __name__ == '__main__':
    app = QApplication([])
    sir = SIRModel()
    sir.show()
    app.exec_()