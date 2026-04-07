# A2S (AdaFusion): Curvature-Aware Adam–SGD Hybrid Optimizer

This project explores how optimizer behavior changes across training phases by blending Adam and SGD with a curvature-aware schedule.

The goal is not to outperform Adam universally, but to study when and why transitioning to SGD-like updates can improve stability and training dynamics.

---

## Core Idea

A2S starts training like Adam for fast convergence, then gradually transitions toward SGD with momentum for improved stability.

A curvature-based scaling term reduces step size when gradients change rapidly, stabilizing updates in high-curvature regions.

### Key Components

- **Adaptive Phase (Early):** Behaves like Adam for rapid progress  
- **Curvature-Aware Scaling:** Shrinks learning rate when gradients fluctuate  
- **Blending Phase:** Gradual transition from Adam → SGD  
- **SGD Phase (Late):** Moves toward momentum-based updates for flatter minima  

---

## ⚙️ Implementation

```python
class A2S:
   
    def __init__(self,
                 lr_adam=0.001,
                 lr_sgd=0.01,
                 beta1=0.9,
                 beta2=0.999,
                 momentum=0.9,
                 curvature_k=0.05,
                 switch_iter=1500,
                 blend_iters=3000,
                 epsilon=1e-8,
                 window_size=10):

        self.lr_adam = lr_adam
        self.lr_sgd = lr_sgd
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.curvature_k = curvature_k
        self.switch_iter = switch_iter
        self.blend_iters = blend_iters
        self.epsilon = epsilon
        self.window_size = window_size

        self.t = 0
        self.m = {}
        self.v = {}
        self.buf = {}
        self.prev_grad = None
        self.grad_history = []

    def update(self, params, grads):
        self.t += 1
        t = self.t

        self.grad_history.append({k: grads[k].copy() for k in grads})
        if len(self.grad_history) > self.window_size:
            self.grad_history.pop(0)

        flat = np.concatenate([g[k].flatten() for g in self.grad_history for k in g])
        grad_variance = np.var(flat) + 1e-12

        if self.prev_grad is None:
            curvature = 0
        else:
            curvature = np.sqrt(sum(
                np.sum((grads[k] - self.prev_grad[k])**2)
                for k in grads
            ))
        self.prev_grad = {k: grads[k].copy() for k in grads}

        lr_scale = 1.0 / (1.0 + self.curvature_k * curvature)
        lr_adam_eff = self.lr_adam * lr_scale
        lr_sgd_eff = self.lr_sgd * lr_scale

        if not self.m:
            for k in params:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
                self.buf[k] = np.zeros_like(params[k])

        for k in params:
            g = grads[k]

            self.m[k] = self.beta1*self.m[k] + (1-self.beta1)*g
            self.v[k] = self.beta2*self.v[k] + (1-self.beta2)*(g*g)

            m_hat = self.m[k] / (1-self.beta1**t)
            v_hat = self.v[k] / (1-self.beta2**t)

            adam_step = lr_adam_eff * m_hat / (np.sqrt(v_hat) + self.epsilon)

            self.buf[k] = self.momentum*self.buf[k] + g
            sgd_step = lr_sgd_eff * self.buf[k]

            if t <= self.switch_iter:
                w = 1.0
            elif t >= self.switch_iter + self.blend_iters:
                w = 0.0
            else:
                w = 1 - (t - self.switch_iter) / self.blend_iters

            params[k] -= w * adam_step + (1-w) * sgd_step

## Experimental Setup

- **Model:** 3-layer MLP (784 → 128 → 64 → 10, ReLU + Softmax)  
- **Dataset:** MNIST  
- **Loss:** Cross-entropy  
- **Batch Size:** 64  
- **Epochs:** 20  

### Baselines
- Adam  
- SGD + Momentum  
- RMSProp  

### Metrics
- Training loss convergence  
- Validation accuracy  
- Stability (loss oscillation)  

---

## Observations

- A2S matches Adam in early convergence speed  
- Transition phase can reduce oscillations in later training  
- Performance is highly sensitive to transition timing and scaling  

---

## Key Insight

The transition schedule (**Adam → SGD**) has a larger impact on training behavior than curvature scaling itself.  

This suggests that **phase-based optimization strategies** may be as important as adaptive learning rate mechanisms.  

---

## Limitations

- Evaluated only on MNIST and shallow networks  
- No consistent improvement over Adam across all settings  
- Hyperparameter sensitivity remains a challenge  

---

## Usage

```python
opt = A2S(lr_adam=0.001, lr_sgd=0.01)

for batch in dataloader:
    preds = model.forward(xb)
    grads = model.backward(yb)
    opt.update(model.params, grads)

## Future Work

- Evaluate on deeper networks (CNNs, CIFAR-10)  
- Incorporate gradient variance into adaptive behavior  
- Improve robustness of transition scheduling  

---

## Summary

A2S is an experimental optimizer combining:

- Fast initial convergence (Adam)  
- Improved stability in later stages (SGD)  
- Curvature-aware step control  

It should be viewed as a research exploration into optimizer dynamics, not a guaranteed replacement for standard optimizers.  

---

## License

Free to use for research, experiments, and academic work.
