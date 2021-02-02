# Runge-Kutta
Generic Runge-Kutta solver (explicit or implicit) and damped Newton method implemented in Python, for educational purposes. A few test cases are given (heat equation, spring-mass system, Curtiss-Hirschfelder).

A script is also provided to study the stability domains, order stars and relative precision contours of any Runge-Kutta scheme.

# TODO:
- time step adaptation:
	- embedded methods where two stage form the solution
	- embedded method where the estimated error has its own coefficients
	- Shampine's trick to L-stabilize the error estimate if necessary
- damped newton improvements:
	- trade between more iterations and more jacobian updates
	- verify scipy's grouped jacobian determination
- stiff PDE test case
