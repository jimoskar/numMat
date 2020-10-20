"""Test numerical gradient on some functions."""
from project2vs import *
"""
f = (lambda x: x**3)
d0 = 1
d = 4
domain = [-2, 2]
I = 500
K = 15
iterations = 2000
chunk = int(I/10)
h = 0.2

NNT = algorithm(I,d,d0, 20,h,iterations, tau, chunk,f,domain,scaling,alpha,beta)

test_input = generate_input(f, domain, d0, I, d)
output, a1, b1, a2, b2 = testing(NNT,test_input,f,domain,d0,d,I,scaling,alpha,beta)
print(output.shape)
grad = NNT.Hamiltonian_gradient()
print(grad.shape)
x = test_input[0,:]
fig, ax = plt.subplots()
ax.scatter(x,grad[0], color="orange", label="Network")
fig.suptitle("Analytical Solution Compared to Output From Network", fontweight = "bold")
ax.set_xlabel("Domain [y]")
ax.set_ylabel("F(y)")

# Plotting the analytical solution
x = np.linspace(domain[0],domain[1])
ax.plot(x,3*x**2, color="blue", label="Function")
ax.legend()
#plt.savefig("compTest1Pic2.pdf", bbox_inches='tight')
plt.show()
"""
#####################################################################
# Test on a function with 2 arguments as well!
f = (lambda x, y: x**2 + y**2)
der = (lambda x, y: 2*x + 2*y)
d0 = 2
d = 4
domain = [[-2,2],[-2,2]]
I = 500
K = 15
iterations = 2000
chunk = int(I/10)
h = 0.2

NN2d = algorithm(I,d,d0, 20,h,iterations, tau, chunk,f,domain,scaling,alpha,beta)

test_input = generate_input(f, domain, d0, I, d)
output, a1, b1, a2, b2 = testing(NN2d,test_input,f,domain,d0,d,I,scaling,alpha,beta)
print(output.shape)
grad = NN2d.Hamiltonian_gradient()
print(grad.shape)

ax = plt.axes(projection='3d')
zdata = grad[0, :] 
xdata = test_input[0,:]
ydata = test_input[1,:]

ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Reds')

# Plotting analytical graph
x = np.linspace(domain[0][0],domain[0][1], 30)
y = np.linspace(domain[1][0], domain[1][1], 30)

X, Y = np.meshgrid(x, y)
Z = der(X, Y)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
        cmap='Greys', edgecolor='none', alpha = 0.5)
plt.show()
