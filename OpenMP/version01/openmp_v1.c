# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>

int main ( );
double *r8vec_linspace_new ( int n, double a, double b );
void timestamp ( );
double *trisolve ( int n, double a[], double b[] );


int main ( ){
  
  double a;
  double *a3;
  double b;
  char command_filename[] = "fd1d_advection_diffusion_steady_commands.txt";
  FILE *command_unit;
  char data_filename[] = "fd1d_advection_diffusion_steady_data.txt";
  FILE *data_unit;
  double dx;
  double *f;
  int i;
  int j;
  double k;
  int nx;
  double r;
  double *u;
  double v;
  double *w;
  double *x;

  printf ( "\n" );
  printf ( "FD1D_ADVECTION_DIFFUSION_STEADY:\n" );
  printf ( "  C version\n" );
  printf ( "\n" );
  printf ( "  Solve the 1D steady advection diffusion equation:,\n" );
  printf ( "    v du/dx - k d2u/dx2 = 0\n" );
  printf ( "  with constant, positive velocity V and diffusivity K\n" );
  printf ( "  over the interval:\n" );
  printf ( "    0.0 <= x <= 1.0\n" );
  printf ( "  with boundary conditions:\n" );
  printf ( "    u(0) = 0, u(1) = 1.\n" );
  printf ( "\n" );
  printf ( "  Use finite differences\n" );
  printf ( "   d u/dx  = (u(t,x+dx)-u(t,x-dx))/2/dx\n" );
  printf ( "   d2u/dx2 = (u(x+dx)-2u(x)+u(x-dx))/dx^2\n" );
  

  // Physical constants.

  v = 1.0;
  k = 0.05;
  printf ( "\n" );
  printf ( "  Diffusivity K = %g\n", k );
  printf ( "  Velocity V    = %g\n", v );

  // Spatial discretization.

  nx = 10000001;
  a = 0.0;
  b = 1.0;
  dx = ( b - a ) / ( double ) ( nx - 1 );

  printf ( "  Number of nodes NX = %d\n", nx );
  printf ( "  DX = %g\n", dx );
  printf ( "  Maximum safe DX is %g\n", 2.0 * k / v / ( b - a ) );

  // Set up the tridiagonal linear system corresponding to the boundary conditions and advection-diffusion equation.

  a3 = ( double * ) malloc ( nx * 3 * sizeof ( double ) );
  f = ( double * ) malloc ( nx * sizeof ( double ) );

  const double start = omp_get_wtime();

  x = r8vec_linspace_new ( nx, a, b );
  
  a3[0+1*nx] = 1.0;
  f[0] = 0.0;

  for ( i = 1; i < nx - 1; i++ )
  {
    a3[i+0*nx] = - v / dx / 2.0 -           k / dx / dx;
    a3[i+1*nx] =                    + 2.0 * k / dx / dx;
    a3[i+2*nx] = + v / dx / 2.0 -           k / dx / dx;
    f[i] = 0.0;
  }

  a3[nx-1+1*nx] = 1.0;
  f[nx-1] = 1.0;

  u = trisolve ( nx, a3, f );

  // The exact solution to the differential equation is known.

  r = v * ( b - a ) / k;

  w = ( double * ) malloc ( nx * sizeof ( double ) );

#pragma omp parallel for
  for ( i = 0; i < nx; i++ )
  {
    w[i] = ( 1.0 - exp ( r * x[i] ) ) / ( 1.0 - exp ( r ) );
  }

  // Write data file.

  data_unit = fopen ( data_filename, "wt" );
  for ( j = 0; j < nx; j++ )
  {
    fprintf ( data_unit, "%g  %g  %g\n", x[j], u[j], w[j] );
  }
  fclose ( data_unit );

  const double end = omp_get_wtime();

  
  printf ( "\n" );
  printf ( "  Gnuplot data written to file '%s'.\n", data_filename );

  // Write command file for visualization (optional).

  command_unit = fopen ( command_filename, "wt" );

  fprintf ( command_unit, "set term png\n" );
  fprintf ( command_unit, "set output 'fd1d_advection_diffusion_steady.png'\n" );
  fprintf ( command_unit, "set grid\n" );
  fprintf ( command_unit, "set style data lines\n" );
  fprintf ( command_unit, "unset key\n" );
  fprintf ( command_unit, "set xlabel '<---X--->'\n" );
  fprintf ( command_unit, "set ylabel '<---U(X)--->'\n" );
  fprintf ( command_unit, "set title 'Exact: green line, Approx: red dots'\n" );
  fprintf ( command_unit, "plot '%s' using 1:2 with points pt 7 ps 2,\\\n", data_filename );
  fprintf ( command_unit, "'' using 1:3 with lines lw 3\n" );
  fprintf ( command_unit, "quit\n" );

  fclose ( command_unit );

  printf ( "  Gnuplot commands written to '%s'\n", command_filename );

  // Free memory.

  free ( a3 );
  free ( f );
  free ( u );
  free ( w );
  free ( x );

  printf("\n  Time taken = %.8f\n", end-start);
  
  // Terminate.

  printf ( "\n" );
  printf ( "FD1D_ADVECTION_DIFFUSION_STEADY\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );

  return 0;
}


double *r8vec_linspace_new ( int n, double a, double b ){
  int i;
  double *x;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  if ( n == 1 )
  {
    x[0] = ( a + b ) / 2.0;
  }
  else
  {
    for ( i = 0; i < n; i++ )
    {
      x[i] = ( ( double ) ( n - 1 - i ) * a 
             + ( double ) (         i ) * b ) 
             / ( double ) ( n - 1     );
    }
  }
  return x;
}



void timestamp (){
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  fprintf ( stdout, "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}



double *trisolve ( int n, double a[], double b[] ){
  int i;
  double *x;
  double xmult;

  // The diagonal entries can't be zero.

  for ( i = 0; i < n; i++ )
  {
    if ( a[i+1*n] == 0.0 )
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "TRISOLVE - Fatal error!\n" );
      fprintf ( stderr, "  A(%d,2) = 0.\n", i );
      exit ( 1 );
    }
  }

  x = ( double * ) malloc ( n * sizeof ( double ) );

  for ( i = 0; i < n; i++ )
  {
    x[i] = b[i];
  }

  for ( i = 1; i < n; i++ )
  {
    xmult = a[i+0*n] / a[i-1+1*n];
    a[i+1*n] = a[i+1*n] - xmult * a[i-1+2*n];
    x[i]   = x[i]   - xmult * x[i-1];
  }

  x[n-1] = x[n-1] / a[n-1+1*n];
  for ( i = n - 2; 0 <= i; i-- )
  {
    x[i] = ( x[i] - a[i+2*n] * x[i+1] ) / a[i+1*n];
  }

  return x;
}
