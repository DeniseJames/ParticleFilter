/*
 * particle_filter.cpp
 *
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include "particle_filter.h"
#include <list>
#include <utility>      // std::pair
#include <math.h>

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Standard deviations for x, y, and theta
	// x is the ground truth plus noise
	// y is the ground truth y plus noise
	// theta is the ground truth theta plus noise
	double std_x, std_y, std_theta;
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// default_random_engine generates a random number from 0.0 to 1.0
	default_random_engine gen;
	// creates a normal (Gaussian) distribution for x, y and theta.
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// Number of particles 
	//num_particles = 10;
	// Resize the particles vector
	particles.resize(num_particles);
	for (int i = 0; i < num_particles; i++)
	{
		particles[i].id = i + 1;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1.0;
	} 

	is_initialized = true;


}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	for (size_t i = 0; i < particles.size(); i++)
	{
		
		particles[i].x = particles[i].x + (velocity/ yaw_rate)*((sin(particles[i].theta+(yaw_rate*delta_t)))-sin(particles[i].theta));
		particles[i].y = particles[i].y + (velocity / yaw_rate)*(cos(particles[i].theta)  - cos(particles[i].theta + (yaw_rate*delta_t)));
		particles[i].theta = particles[i].theta + yaw_rate*delta_t;
		
		// default_random_engine generates a random number from 0.0 to 1.0
		default_random_engine gen;
		// Create a normal (Gaussian) distribution for x, y and theta.
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

		// Assign the uncertainity to particle parts x, y and theta
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);

	} 

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	std::vector<LandmarkObs> observations, Map map_landmarks)
{

	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	// particle weights are set to one
	//std::vector<Particle> predicted_particles;
	//std::vector<Particle> pred_obs_particles;
	std::list<std::pair<LandmarkObs, LandmarkObs>> nearest_neighbor_pair_list;

	// For each particle find the landmarks less than sensor range 

	for (int i = 0; i < particles.size(); i++) {
		std::vector<LandmarkObs> in_range_lm_list;
		// For each map landmark within sensor range to particle 
		// 1. Make list of all landmarks within sensor range of particle, call this `in_range_lm_list`
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			double x_map_lm = map_landmarks.landmark_list[j].x_f;
			double y_map_lm = map_landmarks.landmark_list[j].y_f;
			double dist_particle_to_lm = dist(particles[i].x, particles[i].y, x_map_lm, y_map_lm);
			// if the landmark is less than sensor range then add it to the in_range_lm_list for the particle
			if (dist_particle_to_lm <= sensor_range) {
				// create an object in_range_lm_object to push LandmarkObs onto the in_range_lm_list
				LandmarkObs  in_range_lm_object;
				in_range_lm_object.id = map_landmarks.landmark_list[j].id_i; // original particle id
				in_range_lm_object.x = map_landmarks.landmark_list[j].x_f;  // landmark in range x value
				in_range_lm_object.y = map_landmarks.landmark_list[j].y_f;
				in_range_lm_list.push_back(in_range_lm_object);  // add the in range value to the list
			}   // end of if

			//2. Convert all observations from local to global frame, call this `transformed_obs`
			std::vector<LandmarkObs> transformed_obs;
			for (int k = 0; k < observations.size(); k++) {
				// iterate through all observations and find the shortest distance between
				// the particle and one observation
				// convert observation measurement to map/particle measurement

				LandmarkObs obs_to_map;
				obs_to_map.id = observations[k].id;
				double temp_x = observations[k].x;
				double temp_y = observations[k].y;

				obs_to_map.x = temp_x*cos(particles[i].theta) - temp_y*sin(particles[i].theta) + particles[i].x;
				obs_to_map.y = temp_x*sin(particles[i].theta) + temp_y*cos(particles[i].theta) + particles[i].y;
				transformed_obs.push_back(obs_to_map);
			} // end of k for
			//  3. Perform `dataAssociation`. This will put the index of the `in_range_lm_list` nearest to each 
			//   `transformed_obs` in the `id` field of the `transformed_obs`.
			// make a pair
			for (int m = 0; m < in_range_lm_list.size(); m++) {
				double distance_landmark_to_observation = 50.1;
				for (int n = 0; n < transformed_obs.size(); n++)
				{
					// distance from landmark to observation
					double xd = dist(transformed_obs[n].x, transformed_obs[n].y, in_range_lm_list[m].x, in_range_lm_list[m].y);
					if (xd < distance_landmark_to_observation)
					{	// assign the new shorter distance 
						distance_landmark_to_observation = xd;
						// index of the `in_range_lm_list` nearest to each
						transformed_obs[n].id = in_range_lm_list[m].id;
					}

				}

				for (int p = 0; p < transformed_obs.size(); p++)
				{
					//  4. Loop through all the `transformed_obs`. Use the saved index
					// in the `id` to find the associated landmark and compute the gaussian. 
					for (size_t q = 0; q < in_range_lm_list.size(); q++)
					{
						if (transformed_obs[p].id == in_range_lm_list[q].id)
						{
							// Compute the mult-variate Gaussian distribution of particle
							// pdf is the weight of the particle and the observation
							// calculate the weights
							double C = (1 / (2 * 3.14*std_landmark[0] * std_landmark[1]));
							particles[i].weight = C*exp(-((particles[i].x - transformed_obs[p].x)*(particles[i].x - transformed_obs[p].x) / 2 * std_landmark[0] * std_landmark[0]) +
								(particles[i].y - transformed_obs[p].y)*(particles[i].y - transformed_obs[p].y) / 2 * std_landmark[1] * std_landmark[1]);
						}
					}
				}

			}


		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// particle.weight is the weight vector 
	vector<Particle> resampled_particles;
	vector<double> weights;

	// make a weight vector our of particle weights
	for (size_t i = 0; i < particles.size(); i++)
	{
		double weight_obj = particles[i].weight;
		weights.push_back(weight_obj);
	}

	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end()); 


	int weighted_index = distribution(gen);

	for (int i = 0; i < particles.size(); i++) {
		resampled_particles.push_back(particles[distribution(gen)]);
	}

	particles = resampled_particles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
