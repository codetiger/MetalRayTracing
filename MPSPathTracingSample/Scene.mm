/*
 See LICENSE folder for this sample’s licensing information.
 
 Abstract:
 Implementation for scene creation functions
 */


#import "Scene.h"

using namespace simd;

std::vector<vector_float3> vertices;
std::vector<vector_float3> normals;
std::vector<vector_float2> textureCoords;
std::vector<uint32_t> hasTextures;
std::vector<float> reflections;
std::vector<float> refractions;
std::vector<vector_float3> colors;
std::vector<uint32_t> masks;

float3 getTriangleNormal(float3 v0, float3 v1, float3 v2) {
    float3 e1 = normalize(v1 - v0);
    float3 e2 = normalize(v2 - v0);
    
    return cross(e1, e2);
}

void createCubeFace(std::vector<float3> & vertices,
                    std::vector<float3> & normals,
                    std::vector<float2> & textureCoords,
                    std::vector<float3> & colors,
                    float3 *cubeVertices,
                    float3 color,
                    unsigned int i0,
                    unsigned int i1,
                    unsigned int i2,
                    unsigned int i3,
                    bool inwardNormals,
                    bool hasTexture,
                    float reflection,
                    float refraction,
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

    for (int i = 0; i < 6; i++) {
        reflections.push_back(reflection);
        refractions.push_back(refraction);
        hasTextures.push_back((uint32_t)(hasTexture ? 1 : 0));
    }

    textureCoords.push_back(vector2(0.0f, 0.0f));
    textureCoords.push_back(vector2(0.0f, 1.0f));
    textureCoords.push_back(vector2(1.0f, 1.0f));
    
    textureCoords.push_back(vector2(0.0f, 0.0f));
    textureCoords.push_back(vector2(1.0f, 1.0f));
    textureCoords.push_back(vector2(1.0f, 0.0f));

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
                bool hasTexture,
                float reflection,
                float refraction,
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
        createCubeFace(vertices, normals, textureCoords, colors, cubeVertices, color, 0, 4, 6, 2, inwardNormals, hasTexture, reflection, refraction, triangleMask);
    
    if (faceMask & FACE_MASK_POSITIVE_X)
        createCubeFace(vertices, normals, textureCoords, colors, cubeVertices, color, 1, 3, 7, 5, inwardNormals, hasTexture, reflection, refraction, triangleMask);
    
    if (faceMask & FACE_MASK_NEGATIVE_Y)
        createCubeFace(vertices, normals, textureCoords, colors, cubeVertices, color, 0, 1, 5, 4, inwardNormals, hasTexture, reflection, refraction, triangleMask);
    
    if (faceMask & FACE_MASK_POSITIVE_Y)
        createCubeFace(vertices, normals, textureCoords, colors, cubeVertices, color, 2, 6, 7, 3, inwardNormals, hasTexture, reflection, refraction, triangleMask);
    
    if (faceMask & FACE_MASK_NEGATIVE_Z)
        createCubeFace(vertices, normals, textureCoords, colors, cubeVertices, color, 0, 2, 3, 1, inwardNormals, hasTexture, reflection, refraction, triangleMask);
    
    if (faceMask & FACE_MASK_POSITIVE_Z)
        createCubeFace(vertices, normals, textureCoords, colors, cubeVertices, color, 4, 5, 7, 6, inwardNormals, hasTexture, reflection, refraction, triangleMask);
}

void createSphereTriangle(std::vector<float3> & vertices,
                    std::vector<float3> & normals,
                    std::vector<float2> & textureCoords,
                    std::vector<float3> & colors,
                    float3 *triangleVertices,
                    float3 *triangleNormal,
                    float3 color,
                    bool hasTexture,
                    float reflection,
                    float refraction,
                    unsigned int triangleMask)
{
    for (int i = 0; i < 3; i++) {
        reflections.push_back(reflection);
        refractions.push_back(refraction);
        hasTextures.push_back((uint32_t)(hasTexture ? 1 : 0));
    }
    
    textureCoords.push_back(vector2(0.0f, 0.0f));
    textureCoords.push_back(vector2(0.0f, 1.0f));
    textureCoords.push_back(vector2(1.0f, 1.0f));
    
    for (int i = 0; i < 3; i++) {
        vertices.push_back(triangleVertices[i]);
        normals.push_back(triangleNormal[i]);
        colors.push_back(color);
    }
    
    masks.push_back(triangleMask);
}

#define RING_STEPS 30
#define POINT_STEPS 30

void createSphere(vector_float3 color,
                  matrix_float4x4 transform,
                  bool hasTexture,
                  float reflection,
                  float refraction,
                  unsigned int triangleMask) {
    float pi = 22.0f / 7.0f;
    float deltaTheta = pi / (RING_STEPS + 2);
    float deltaPhi = 2.0f * pi / POINT_STEPS;
    float theta = 0.0f;
    
    for (int ring = 0; ring < RING_STEPS + 2; ring++) {
        float phi = 0.0f;
        for (int point = 0; point < POINT_STEPS; point++) {
            float3 v0 = vector3(sin(theta) * cos(phi),
                               sin(theta) * sin(phi),
                               cos(theta));

            float3 v1 = vector3(sin(theta + deltaTheta) * cos(phi),
                               sin(theta + deltaTheta) * sin(phi),
                               cos(theta + deltaTheta));

            float3 v2 = vector3(sin(theta) * cos(phi + deltaPhi),
                               sin(theta) * sin(phi + deltaPhi),
                                cos(theta));
            
            float3 v3 = vector3(sin(theta + deltaTheta) * cos(phi + deltaPhi),
                               sin(theta + deltaTheta) * sin(phi + deltaPhi),
                                cos(theta + deltaTheta));
            
            float3 triangleVertices0[] = {
                (transform * vector4(v0, 1.0f)).xyz,
                (transform * vector4(v1, 1.0f)).xyz,
                (transform * vector4(v2, 1.0f)).xyz };

            float3 normal0[] = {
                (transform * vector4(v0, 0.0f)).xyz,
                (transform * vector4(v1, 0.0f)).xyz,
                (transform * vector4(v2, 0.0f)).xyz};
            createSphereTriangle(vertices, normals, textureCoords, colors, triangleVertices0, normal0, color, hasTexture, reflection, refraction, triangleMask);
            
            float3 triangleVertices1[] = {
                (transform * vector4(v3, 1.0f)).xyz,
                (transform * vector4(v2, 1.0f)).xyz,
                (transform * vector4(v1, 1.0f)).xyz};

            float3 normal1[] = {
                (transform * vector4(v3, 0.0f)).xyz,
                (transform * vector4(v2, 0.0f)).xyz,
                (transform * vector4(v1, 0.0f)).xyz};
            createSphereTriangle(vertices, normals, textureCoords, colors, triangleVertices1, normal1, color, hasTexture, reflection, refraction, triangleMask);
            phi += deltaPhi;
        }
        theta += deltaTheta;
    }
    
//    drawVertex(0,0,1) //north pole end cap
//    drawVertex(0, 0, -1) //south pole end cap
}
