# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <mpi.h>

int main ( );
double *r8vec_linspace_new ( int n, double a, double b );
void timestamp ( );
double *trisolve ( int n, double a[], double b[] );


int main ( ){

  MPI_Init(NULL, NULL);
  MPI_Status status;
  int rank,world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);
  
  
  double a;
  double *a3;
  double b;
  char command_filename[] = "fd1d_advection_diffusion_steady_commands.txt";
  FILE *command_unit;
  char data_filename[] = "fd1d_advection_diffusion_steady_data.txt";
  FILE *data_unit;
  double dx;
  double *f;
  double *f_dup;
  int i;
  int j;
  double k;
  int nx;
  double r;
  double *u;
  double v;
  double *w;
  double *x;
  int segment_size, segment_start, segment_end;

  const int root = 0;
 
  // Physical constants.
  v = 1.0;
  k = 0.05;
  
  // Spatial discretization.
  nx = 10000001;
  a = 0.0;
  b = 1.0;
  dx = ( b - a ) / ( double ) ( nx - 1 );

  //Set up the tridiagonal linear system corresponding to the boundary conditions and advection-diffusion equation.
  a3 = ( double * ) malloc ( nx * 3 * sizeof ( double ) );
  f = ( double * ) malloc ( nx * sizeof ( double ) );

  segment_size = nx/world_size;
  segment_start = rank * segment_size;
  segment_end = (rank+1) * segment_size;
 

  const double start = MPI_Wtime();

  x = r8vec_linspace_new ( nx, a, b );
  
  if(rank == root){
    for (int i = segment_start; i < segment_end; i++ ){
      a3[i+0*nx] =- v / dx / 2.0 - k / dx / dx;
      a3[i+1*nx] =+ 2.0 * k / dx / dx;
      a3[i+2*nx] =+ v / dx / 2.0 - k / dx / dx;
      f[i] = 0.0;
    }
   

    MPI_Ssend(a3, nx*3, MPI_DOUBLE, rank+1, 0,MPI_COMM_WORLD);
    MPI_Ssend(f, nx, MPI_DOUBLE, rank+1, 0,MPI_COMM_WORLD);


    MPI_Recv(a3, nx*3, MPI_DOUBLE, world_size-1, 0,MPI_COMM_WORLD, &status);
    MPI_Recv(f, nx, MPI_DOUBLE, world_size-1, 0,MPI_COMM_WORLD, &status);

    a3[0+1*nx] = 1.0;
    a3[nx-1+1*nx] = 1.0;
    f[0] = 0.0;
   
    

  }
  else if(rank == world_size-1){
    
    MPI_Recv(a3, nx*3, MPI_DOUBLE, rank-1, 0,MPI_COMM_WORLD, &status);
    MPI_Recv(f, nx, MPI_DOUBLE, rank-1, 0,MPI_COMM_WORLD, &status);
    

    for ( i = segment_start; i < segment_end; i++ ){
      a3[i+0*nx] =- v / dx / 2.0 - k / dx / dx;
      a3[i+1*nx] =+ 2.0 * k / dx / dx;
      a3[i+2*nx] =+ v / dx / 2.0 - k / dx / dx;
      f[i] = 0.0;
    }

    MPI_Ssend(a3, nx*3, MPI_DOUBLE, root, 0,MPI_COMM_WORLD);
    MPI_Ssend(f, nx, MPI_DOUBLE, root, 0,MPI_COMM_WORLD);
   
  }
  else{
    MPI_Recv(a3, nx*3, MPI_DOUBLE, rank-1, 0,MPI_COMM_WORLD, &status);
    MPI_Recv(f, nx, MPI_DOUBLE, rank-1, 0,MPI_COMM_WORLD, &status);
    
    for ( i = segment_start; i < segment_end; i++ ){
      a3[i+0*nx] =- v / dx / 2.0 - k / dx / dx;
      a3[i+1*nx] =+ 2.0 * k / dx / dx;
      a3[i+2*nx] =+ v / dx / 2.0 - k / dx / dx;
      f[i] = 0.0;
    }

    MPI_Ssend(a3, nx*3, MPI_DOUBLE, rank+1, 0,MPI_COMM_WORLD);
    MPI_Ssend(f, nx, MPI_DOUBLE, rank+1, 0,MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(a3, nx*3,MPI_DOUBLE,root,MPI_COMM_WORLD);
  MPI_Bcast(f,nx,MPI_DOUBLE,root,MPI_COMM_WORLD);


  MPI_Barrier(MPI_COMM_WORLD);
 

  u = trisolve ( nx, a3, f );

  // The exact solution to the differential equation is known.

  r = v * ( b - a ) / k;
  w = ( double * ) malloc ( nx * sizeof ( double ) );

  for ( i = 0; i < nx; i++ )
  {
    w[i] = ( 1.0 - exp ( r * x[i] ) ) / ( 1.0 - exp ( r ) );
  }

  // Write data file.
  if( rank == world_size-1){
    data_unit = fopen ( data_filename, "wt" );
    for ( j = 0; j < nx; j++ )
      {
	fprintf ( data_unit, "%g  %g  %g\n", x[j], u[j], w[j] );
      }
    fclose ( data_unit );
  }

  const double end = MPI_Wtime();

  
  printf ( "\n" );
  printf ( "  Gnuplot data written to file '%s'.\n", data_filename );

  // Write command file.

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

  MPI_Finalize();
  
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



void timestamp ( ){
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
