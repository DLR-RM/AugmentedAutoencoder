#version 450 core

in vec3 v_color;
in vec3 v_view_z;

in vec3 v_normal;
in vec3 v_light_dir;

layout (location = 0) out vec3 rgb;
layout (location = 1) out float depth;


void main(void) {
    vec3 Normal = normalize(v_normal);
    vec3 LightDir = normalize(v_light_dir);
    vec3 ViewDir = normalize(v_view_z);

    vec3 material_ambient = vec3(223./255, 214./255, 205./255 );
    vec3 material_diffuse = vec3(223./255, 214./255, 205./255 );
    vec3 material_specular = vec3(223./255, 214./255  , 205./255 );
    vec3 diffuse = max(dot(Normal, LightDir), 0.0) * material_diffuse;
    vec3 R = reflect(-LightDir, Normal);
    vec3 specular = pow(max(dot(R, ViewDir), 0.0), 0.00* 128.0) * material_specular;

    vec3 dirlight_ambient = vec3(0.4);
    vec3 dirlight_diffuse = vec3(0.7);
    vec3 dirlight_specular = vec3(1.0);

    rgb = vec3(     dirlight_ambient  * material_ambient + 
                    dirlight_diffuse  * diffuse); // + 
                   // dirlight_specular *specular);
    
	depth = v_view_z.z;
}




