
def euler_forward_step(f, x, u, p, dt):
    return x + dt * f(x,u,p)




class RKSimulator:
    def __init__(self, step_size, num_steps):
        self.dt = step_size
        self.N = num_steps

    def make_step(self, f, x0, u, p):
   
        x = x0
        for i in range(self.N):
            k1 = f(x, u, p)
            k2 = f(x + 0.5 * self.dt * k1, u, p)
            k3 = f(x + 0.5 * self.dt * k2, u, p)
            k4 = f(x + self.dt * k3, u, p)
            x = x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return x