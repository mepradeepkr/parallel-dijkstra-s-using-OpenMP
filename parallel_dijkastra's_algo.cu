#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

#define N 16
#define SOURCE 0
#define MAXINT 999999

void print_path(int source, int dest, int parent[], int distance[], int graph[][N]) {
    int curr = dest;
    printf("%d: ",distance[curr]);

    while(curr != source) {
        printf("%d(%d) ---> ",curr,graph[curr][parent[curr]]);
        curr = parent[curr];
    }
    printf("%d\n",curr);
}


void dijkstra(int graph[N][N], int source);

void dijkstra(int graph[N][N], int source){
    
    int i;

    int visited[N];
    int distance[N]; 
    int parent[N];

    for(int i=0; i<N; i++) {
        parent[i] = -1;
    }

    int global_min_dist;
    int global_min_vertex;

    int th_id;    
    int th_start_idx; //first vertex index that stores in one thread locally
    int th_last_idx;  //last vertex index that stores in one thread locally

    int loc_min_dist;    //local minimum distance
    int loc_min_vertex;   //local minimum vertex

    int my_step;  
    int nth;      //total number of threads

    //Initialize all vertices' distance and visited status. 
    for (i = 0; i < N; i++){
        visited[i] = 0;
        distance[i] = graph[source][i];
        if(graph[source][i] < MAXINT) {
            parent[i] = source;
        }
    }
    visited[source] = 1;
    
    #pragma omp parallel private(th_start_idx, th_id, th_last_idx, loc_min_dist, loc_min_vertex, my_step) shared(visited, global_min_dist, distance, global_min_vertex, nth, graph)
    {
        th_id = omp_get_thread_num();
        nth = omp_get_num_threads();

        th_start_idx = (th_id * N) / nth;
        th_last_idx = ((th_id + 1) * N) / nth - 1;

        // printf(stdout, "P%d: First=%d Last=%d\n", th_id, th_start_idx, th_last_idx);

        for (my_step = 1; my_step < N; my_step++){

            #pragma omp single
            {
                global_min_dist = MAXINT;
                global_min_vertex = -1;
            }

            int vertex_k;
            
            loc_min_dist = MAXINT;
            loc_min_vertex = -1;

            //each thread will find out local_min in distance[] array and corresponding vertex index in its local indices range.
            for (vertex_k = th_start_idx; vertex_k <= th_last_idx; vertex_k++){
                if (!visited[vertex_k] && distance[vertex_k] < loc_min_dist){
                    loc_min_dist = distance[vertex_k];
                    loc_min_vertex = vertex_k;
                }
            }


            //find out the value of distance which is minimum globally and corresponding vertex index
            #pragma omp critical
            {
                if (loc_min_dist < global_min_dist){
                    global_min_dist = loc_min_dist;
                    global_min_vertex = loc_min_vertex;
                }
            }
            #pragma omp barrier

            
            //now, global_min_vertex is visited
            #pragma omp single
            {
                if (global_min_vertex != -1){
                    visited[global_min_vertex] = 1;
                }
            }


            //each thread will update distance value for each vertex after visiting global_min_vertex
            #pragma omp barrier
            if (global_min_vertex != -1){
                for (int j = th_start_idx; j <= th_last_idx; j++){
                    if (!visited[j] && graph[global_min_vertex][j] < MAXINT && distance[global_min_vertex] + graph[global_min_vertex][j] < distance[j]){
                        printf("Updating %d to %d\n",j,global_min_vertex);
                        parent[j] = global_min_vertex;
                        distance[j] = distance[global_min_vertex] + graph[global_min_vertex][j];
                    }
                }
            }
            #pragma omp barrier
        }
    }

    printf("\nDistance vector: \n");
    for (i = 0; i < N; i++) {
        print_path(source, i, parent, distance, graph);
    }
}

void genrate_array(int a[N][N]){
    int n = N;
    for (int i = 0; i < n; i++){
        a[i][i] = 0;
        for(int j = 0; j < i; j++){
            if(rand() % 2 == 0) {
                a[i][j] = MAXINT;
            }
            else {
                a[i][j] = (rand() % 10) + 5;
            }
            a[j][i] = a[i][j];
        }
    }
}

void printarray(int a[N][N]){
    int n = N;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }
}



int main(int argc, char **argv){
    int i, j;
    int graph[N][N];

//    int graph[N][N] = { { 0, 4, 0, 0, 0, 0, 0, 8, 0 },
//                         { 4, 0, 8, 0, 0, 0, 0, 11, 0 },
//                         { 0, 8, 0, 7, 0, 4, 0, 0, 2 },
//                         { 0, 0, 7, 0, 9, 14, 0, 0, 0 },
//                         { 0, 0, 0, 9, 0, 10, 0, 0, 0 },
//                         { 0, 0, 4, 14, 10, 0, 2, 0, 0 },
//                         { 0, 0, 0, 0, 0, 2, 0, 1, 6 },
//                         { 8, 11, 0, 0, 0, 0, 1, 0, 7 },
//                         { 0, 0, 2, 0, 0, 0, 6, 7, 0 } };

    // for(int i=0; i<N; i++) {
    //     for(int j=0; j<N; j++) {
    //         if(graph[i][j] == 0) {
    //             graph[i][j] = MAXINT;
    //         } 
    //     }
    //     graph[i][i] = 0;
    // }


    int threads;

    printf("Enter number of threads: ");
    scanf("%d", &threads);

    omp_set_num_threads(threads);

    double time_start, time_end;

    struct timeval tv;
    struct timezone tz;

    gettimeofday(&tv, &tz);

    time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;



    printf("\nAdjacency matrix: \n");

    genrate_array(graph);
    printarray(graph);
    dijkstra(graph, SOURCE);

    gettimeofday(&tv, &tz);

    time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.00;
    printf("Nodes: %d\n", N);

    printf("time cost is %1f\n", time_end - time_start);
    return 0;
}
