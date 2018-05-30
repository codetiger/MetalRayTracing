/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Implementation for scene creation functions
*/


#import "Scene.h"

using namespace simd;

std::vector<vector_float3> vertices;
std::vector<vector_float3> normals;
std::vector<vector_float3> colors;
std::vector<uint32_t> masks;

float3 getTriangleNormal(float3 v0, float3 v1, float3 v2) {
    float3 e1 = normalize(v1 - v0);
    float3 e2 = normalize(v2 - v0);
    
    return cross(e1, e2);
}

void createCubeFace(std::vector<float3> & vertices,
                    std::vector<float3> & normals,
                    std::vector<float3> & colors,
                    float3 *cubeVertices,
                    float3 color,
                    unsigned int i0,
                    unsigned int i1,
                    unsigned int i2,
                    unsigned int i3,
                    bool inwardNormals,
                    unsigned int triangleMask)
{
    float3 v0 = cubeVertices[i0];
    float3 v1 = cubeVertices[i1];
    float3 v2 = cubeVertices[i2];
    float3 v3 = cubeVertices[i3];
    
    float3 n0 = getTriangleNormal(v0, v1, v2);
    float3 n1 = getTriangleNormal(v0, v2, v3);
    
    if (inwardNormals) {
        n0 = -n0;
        n1 = -n1;
    }
    
    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
    vertices.push_back(v0);
    vertices.push_back(v2);
    vertices.push_back(v3);
    
    for (int i = 0; i < 3; i++)
        normals.push_back(n0);
    
    for (int i = 0; i < 3; i++)
        normals.push_back(n1);
    
    for (int i = 0; i < 6; i++)
        colors.push_back(color);
    
    for (int i = 0; i < 2; i++)
        masks.push_back(triangleMask);
}

void createCube(unsigned int faceMask,
                vector_float3 color,
                matrix_float4x4 transform,
                bool inwardNormals,
                unsigned int triangleMask)
{
    float3 cubeVertices[] = {
        vector3(-0.5f, -0.5f, -0.5f),
        vector3( 0.5f, -0.5f, -0.5f),
        vector3(-0.5f,  0.5f, -0.5f),
        vector3( 0.5f,  0.5f, -0.5f),
        vector3(-0.5f, -0.5f,  0.5f),
        vector3( 0.5f, -0.5f,  0.5f),
        vector3(-0.5f,  0.5f,  0.5f),
        vector3( 0.5f,  0.5f,  0.5f),
    };
    
    for (int i = 0; i < 8; i++) {
        float3 vertex = cubeVertices[i];
        
        float4 transformedVertex = vector4(vertex.x, vertex.y, vertex.z, 1.0f);
        transformedVertex = transform * transformedVertex;
        
        cubeVertices[i] = transformedVertex.xyz;
    }
    
    if (faceMask & FACE_MASK_NEGATIVE_X)
        createCubeFace(vertices, normals, colors, cubeVertices, color, 0, 4, 6, 2, inwardNormals, triangleMask);
    
    if (faceMask & FACE_MASK_POSITIVE_X)
        createCubeFace(vertices, normals, colors, cubeVertices, color, 1, 3, 7, 5, inwardNormals, triangleMask);
    
    if (faceMask & FACE_MASK_NEGATIVE_Y)
        createCubeFace(vertices, normals, colors, cubeVertices, color, 0, 1, 5, 4, inwardNormals, triangleMask);
    
    if (faceMask & FACE_MASK_POSITIVE_Y)
        createCubeFace(vertices, normals, colors, cubeVertices, color, 2, 6, 7, 3, inwardNormals, triangleMask);
    
    if (faceMask & FACE_MASK_NEGATIVE_Z)
        createCubeFace(vertices, normals, colors, cubeVertices, color, 0, 2, 3, 1, inwardNormals, triangleMask);
    
    if (faceMask & FACE_MASK_POSITIVE_Z)
        createCubeFace(vertices, normals, colors, cubeVertices, color, 4, 5, 7, 6, inwardNormals, triangleMask);
}
