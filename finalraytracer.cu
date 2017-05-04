#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
//#include <ctime>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define true            1
#define false           0
//#define M_PI            3.141592653589793
//#define INFINITY        1e8

#define MAX_RAY_DEPTH   5
#define NO_OF_SPHERES   5
#define NO_OF_LIGHTS    1

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef struct vec
{
    float x;
    float y;
    float z;
}vec3;

typedef struct Sphere_t
{
    vec3  center;
    float radius;
    float radius2;
    vec3  surface_color;
    float reflection;
    float transparency;
    vec3  emission_color;
} sphere_t;

__device__ void vec3_init(vec3 *name, float a, float b, float c)
{
    name->x = a;
    name->y = b;
    name->z = c;
}

__device__ vec3 vec3_normalize(vec3 name)
{
    float sq, invsqrt;
    vec3 op;

    sq = name.x * name.x + name.y * name.y + name.z * name.z;
    invsqrt = 1 / sqrt(sq);
    op.x = name.x * invsqrt;
    op.y = name.y * invsqrt;
    op.z = name.z * invsqrt;

    return op;
}

__device__ vec3 vec3_add(vec3 op1, vec3 op2)
{
    vec3 dest;

    dest.x = op1.x + op2.x;
    dest.y = op1.y + op2.y;
    dest.z = op1.z + op2.z;

    return dest;
}

__device__ vec3 vec3_sub(vec3 op1, vec3 op2)
{
    vec3 dest;

    dest.x = op1.x - op2.x;
    dest.y = op1.y - op2.y;
    dest.z = op1.z - op2.z;

    return dest;
}

__device__ vec3 vec3_mul(vec3 op1, vec3 op2)
{
    vec3 dest;

    dest.x = op1.x * op2.x;
    dest.y = op1.y * op2.y;
    dest.z = op1.z * op2.z;

    return dest;
}

__device__ vec3 vec3_const_mul(vec3 ip, float value)
{
    vec3 op;

    op.x = ip.x * value;
    op.y = ip.y * value;
    op.z = ip.z * value;

    return op;
}

__device__ float vec3_dot(vec3 *op1, vec3 *op2)
{
    return (((op1->x)*(op2->x)) + ((op1->y)*(op2->y)) + ((op1->z)*(op2->z)));
}

__device__ vec3 vec3_negate(vec3 ip)
{
    vec3 op;

    op.x = -ip.x;
    op.y = -ip.y;
    op.z = -ip.z;

    return op;
}

__device__ void vec3_copy(vec3 *dest, vec3 *source)
{
    dest->x = source->x;
    dest->y = source->y;
    dest->z = source->z;
}

__device__ bool intersect(vec3 ray_origin, vec3 ray_dir, sphere_t s, float *t0, float *t1)
{
    vec3 l;
    l = vec3_sub(s.center, ray_origin);
    float tca = vec3_dot(&l, &ray_dir);
    if (tca < 0)
        return false;
    float d2 = vec3_dot(&l, &l);
    d2 -= (tca*tca);
    if (d2 > s.radius2)
        return false;
    float thc = sqrt(s.radius2 - d2);
    *t0 = tca - thc;
    *t1 = tca + thc;

    //free all the allocated temp variables

    return true;
}

__device__ float mix(float a, float b, float mix)
{
    return (b * mix + a * (1 - mix));
}


__global__ void render(sphere_t *spheres, int no_of_spheres, 
                        int no_of_lights, float invWidth, float invHeight, 
                        float aspectratio, float angle, vec3 *image_output, 
                        unsigned int width, unsigned int height, int count,int ray_depth)
//__global__ void render(vec3 *image_output, int width, int height)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*width + ix;
    unsigned int y;
    unsigned int x;
    float xx;
    float yy;
	int ik = 0;

    
	for (ik = 0; ik < count; ik++)
{
    if (ix < width && iy < height)
    {
       
        
        y = idx / width;
        x = idx % width;
        xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio;
        yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
        
        vec3 raydir;
        vec3_init(&raydir, xx, yy, -1);
        raydir = vec3_normalize(raydir);
        vec3 rayorig;
        vec3_init(&rayorig, 0, 0, 0);
        
        vec3 final_color;
        vec3 last_reflection;
        vec3 scale;
        vec3 nhit;
        vec3 phit;
        vec3 surfaceColor;

        float    bias = 1e-4;
        float    tnear = INFINITY;
        sphere_t *sphere = NULL;
        int      depth = 0;

        vec3_init(&scale, 1, 1, 1);
        vec3_init(&final_color, 0, 0, 0);
        vec3_init(&surfaceColor, 0, 0, 0);
        vec3_init(&last_reflection, 0, 0, 0);

        do
        {
            sphere = NULL;
            for (unsigned i = 0; i < no_of_spheres + no_of_lights; ++i)
            {
                float t0 = INFINITY, t1 = INFINITY;
                if (intersect(rayorig, raydir, spheres[i], &t0, &t1))
                {
                    if (t0 < 0) t0 = t1;
                    if (t0 < tnear)
                    {
                        tnear = t0;
                        sphere = &spheres[i];
                    }
                }
            }
            // if there's no intersection return black or background color
            if (!sphere)
            {
                vec3_init(&last_reflection, 2, 2, 2);
                break;
            }

            vec3 temp2;
            temp2 = vec3_const_mul(raydir, tnear);
            phit = vec3_add(rayorig, temp2);

            nhit = vec3_sub(phit, sphere->center);

            nhit = vec3_normalize(nhit);

            // If the normal and the view direction are not opposite to each other
            // reverse the normal direction. That also means we are inside the sphere so set
            // the inside bool to true. Finally reverse the sign of IdotN which we want
            // positive.

            if (vec3_dot(&raydir, &nhit) > 0)
            {
                nhit = vec3_negate(nhit);
            }

            if (sphere->reflection > 0 && depth < ray_depth)
            {
                float facingratio = -(vec3_dot(&raydir, &nhit));
                // change the mix value to tweak the effect
                float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
                // compute reflection direction (not need to normalize because all vectors
                // are already normalized)
                vec3 refldir = vec3_sub(raydir, vec3_const_mul(nhit, 2 * vec3_dot(&raydir, &nhit)));

                refldir = vec3_normalize(refldir);
                vec3 temp10, temp11;

                temp10 = vec3_const_mul(nhit, bias);
                temp11 = vec3_add(phit, temp10);

                vec3_copy(&rayorig, &temp11);
                vec3_copy(&raydir, &refldir);

                final_color = vec3_add(final_color, vec3_mul(sphere->emission_color, scale));
                scale = vec3_mul(scale, vec3_const_mul(sphere->surface_color, fresneleffect));
            }
            else
            {
                for (unsigned i = no_of_spheres; i < no_of_spheres + no_of_lights; ++i)
                {
                    // this is a light
                    vec3 transmission;
                    vec3_init(&transmission, 1, 1, 1);
                    vec3 lightDirection = vec3_sub(spheres[i].center, phit);
                    lightDirection = vec3_normalize(lightDirection);

                    for (unsigned j = 0; j < no_of_spheres + no_of_lights; ++j)
                    {
                        float t0 = 0, t1 = 0;
                        vec3 temp3;
                        temp3 = vec3_add(phit, vec3_const_mul(nhit, bias));

                        if (intersect(temp3, lightDirection, spheres[j], &t0, &t1))
                        {
                            vec3_init(&transmission, 0, 0, 0);
                            break;
                        }
                    }
                    vec3 temp8, temp9;
                    float ftemp1;

                    temp8 = vec3_mul(sphere->surface_color, transmission);

                    ftemp1 = MAX((float)0, vec3_dot(&nhit, &lightDirection));

                    temp9 = vec3_const_mul(spheres[i].emission_color, ftemp1);

                    temp9 = vec3_mul(temp8, temp9);

                    surfaceColor = vec3_add(surfaceColor, temp9);
                }
                last_reflection = vec3_add(surfaceColor, sphere->emission_color);
            }
            depth++;
        } while (depth < ray_depth + 1);

        final_color = vec3_add(final_color, vec3_mul(last_reflection, scale));

        image_output[idx].x = final_color.x;
        image_output[idx].y = final_color.y;
        image_output[idx].z = final_color.z;
    }
}
}


void init_spheres(sphere_t *sphere,
    float posx, float posy, float posz,
    float radius,
    float surfx, float surfy, float surfz,
    float reflection,
    float transparency,
    float emisx, float emisy, float emisz
    )
{
    sphere->center.x = posx;
    sphere->center.y = posy;
    sphere->center.z = posz;

    sphere->radius = radius;
    sphere->radius2 = sphere->radius * sphere->radius;

    sphere->surface_color.x = surfx;
    sphere->surface_color.y = surfy;
    sphere->surface_color.z = surfz;

    sphere->reflection = reflection;
    sphere->transparency = transparency;

    sphere->emission_color.x = emisx;
    sphere->emission_color.y = emisy;
    sphere->emission_color.z = emisz;
}



int main(int argc, char **argv)
{
    sphere_t    *spheres;
    sphere_t    *d_spheres;

    vec3        *h_image_output;
    vec3        *d_image_output;

    int         no_of_spheres;
    int         no_of_lights;
	
	clock_t begin, end;
	double time_spent;
 int ray_depth;
 

    int count;
    
    count = atoi(argv[1]);
    ray_depth = atoi(argv[2]);
    
    if (!ray_depth)
    {
      ray_depth = 5;
      printf("ray depth is default value i.e., 5 \n");
    }
  
    unsigned int width = atoi(argv[3]);
    unsigned int height = atoi(argv[4]);

    float       invWidth = 1 / (float)width;
    float       invHeight = 1 / (float)height;
    float       fov = 30;
    float       aspectratio = width / (float)height;
    float       angle = tan(M_PI * 0.5 * fov / 180.);

    no_of_spheres = NO_OF_SPHERES;
    no_of_lights = NO_OF_LIGHTS;

    spheres = (sphere_t *)calloc((no_of_spheres + no_of_lights), sizeof(sphere_t));
    h_image_output = (vec3 *)calloc(width * height, sizeof(vec3));

    if (cudaMalloc(&d_spheres, sizeof(sphere_t) * (no_of_spheres + no_of_lights)) != cudaSuccess)
    {
        printf("Memory allocation failed for d_spheres\n");
        return 0;
    }

    if (cudaMalloc(&d_image_output, sizeof(vec3) * width * height) != cudaSuccess)
    {
        printf("Memory allocation failed during d_image_output \n");
        return 0;
    }

    init_spheres(&spheres[0],
        0.0, -10004, -20,
        10000,
        0.20, 0.20, 0.20,
        0,
        0.0,
        0.0, 0.0, 0.0);

    init_spheres(&spheres[1],
        0.0, 0, -20,
        4,
        1.00, 0.32, 0.36,
        1,
        0.5,
        0.0, 0.0, 0.0);

    init_spheres(&spheres[2],
        5.0, -1, -15,
        2,
        0.90, 0.76, 0.46,
        1,
        0.0,
        0.0, 0.0, 0.0);

    init_spheres(&spheres[3],
        5.0, 0, -25,
        3,
        0.65, 0.77, 0.97,
        1,
        0.0,
        0.0, 0.0, 0.0);

    init_spheres(&spheres[4],
        -5.5, 0, -15,
        3,
        0.90, 0.90, 0.90,
        1,
        0.0,
        0.0, 0.0, 0.0);

    // light
    
    init_spheres(&spheres[5],
        0.0, 20, -30,
        3,
        0.00, 0.00, 0.00,
        1,
        0.0,
        3, 0.0, 0.0);

    if (cudaMemcpy(d_spheres, spheres, sizeof(sphere_t) * (no_of_spheres + no_of_lights), cudaMemcpyHostToDevice) != cudaSuccess)
    {
        printf("Data transfer of d_a from host to device failed");
        free(spheres);
        cudaFree(d_spheres);
        return 0;
    }
  begin = clock();

    dim3 block(32, 32);
    dim3 grid((width+block.x -1)/block.x, (height + block.y - 1)/block.y);
	
	

    render <<<grid, block>>>(d_spheres, no_of_spheres, no_of_lights, 
                            invWidth, invHeight, aspectratio, angle, 
                            d_image_output, width, height, count, ray_depth);

    cudaDeviceSynchronize();
end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	
	printf("the execution time of kernel is %f \n",(time_spent));
	
    if (cudaMemcpy(h_image_output, d_image_output, sizeof(vec3) * width * height, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        printf("Data transfer of d_image_output from device to host failed \n");
        free(h_image_output);
        cudaFree(d_image_output);
        return 0;
    }


	
    FILE *fp = fopen("first.ppm", "wb"); /* b - binary mode */
    FILE *fp1 = fopen("first1.txt", "w+"); /* b - binary mode */
    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    for (unsigned i = 0; i < width * height; ++i)
    {
        static unsigned char color[3];
        color[0] = MIN((float)1, h_image_output[i].x) * 255;
        color[1] = MIN((float)1, h_image_output[i].y) * 255;
        color[2] = MIN((float)1, h_image_output[i].z) * 255;

        (void)fwrite(color, 1, 3, fp);
        fprintf(fp1, "%f, %f, %f\n", h_image_output[i].x, h_image_output[i].y, h_image_output[i].z);

    }
    fclose(fp);
    fclose(fp1);

    free(h_image_output);
    free(spheres);

    cudaFree(d_image_output);
    cudaFree(d_spheres);



    return 0;
}
