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
#include <cassert>

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
	num_particles = 50;
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

static vector<LandmarkObs> STransformObservationToMapCoordinates(const vector<LandmarkObs>& o, const Particle& p)
{
	// translate the observations to map space
	std::vector<LandmarkObs> transformed_o;
	for (int k = 0; k < o.size(); k++) {
		// iterate through all observations and find the shortest distance between
		// the particle and one observation
		// convert observation measurement to map/particle measurement

		LandmarkObs obs_to_map;
		obs_to_map.id = o[k].id;
		double temp_x = o[k].x;
		double temp_y = o[k].y;

		obs_to_map.x = temp_x*cos(p.theta) - temp_y*sin(p.theta) + p.x;
		obs_to_map.y = temp_x*sin(p.theta) + temp_y*cos(p.theta) + p.y;
		transformed_o.push_back(obs_to_map);
	 }
	  return transformed_o;
}

//  3. Perform `dataAssociation`. This will put the index of the landmark `in_range_lm_list`
// nearest to each  `transformed_obs` in the `id` field of the `transformed_obs`.
// 
static std::list<std::pair<LandmarkObs, LandmarkObs>>  SDataAssociation(vector<LandmarkObs> r, vector<LandmarkObs>& to)
{
	vector<LandmarkObs> DataAssociation;
	std::list<std::pair<LandmarkObs, LandmarkObs>> nn_pair_list;
	for (int m = 0; m < r.size(); m++) 
	{
		LandmarkObs landmark = r[m];
		
		LandmarkObs closest_observation;
		double distance_landmark_to_observation = 50.1;
		for (int n = 0; n < to.size(); n++)
		{
			LandmarkObs transformed_observation = to[n];
			// distance from predicted landmark to observation
			double xd = dist(transformed_observation.x, transformed_observation.y, landmark.x, landmark.y);
			if (xd < distance_landmark_to_observation)
			{	// assign the new shorter distance 
				distance_landmark_to_observation = xd;
				// make an object to push onto the Dataassociation list
				closest_observation.id = transformed_observation.id;
				closest_observation.x = transformed_observation.x;
				closest_observation.y = transformed_observation.y;
				// index of the `in_range_lm_list` assigned to transformed obs
			}
			
		}
		std::pair<LandmarkObs, LandmarkObs> obs_lm_pair;
		// make a pair of closest observation and landmark
		obs_lm_pair = std::make_pair(landmark, closest_observation);
		// put the pair on the list
		nn_pair_list.push_back(obs_lm_pair);
		
	}
	
	return nn_pair_list;
}

static double WeightParticleFromNearestObservation(const std::list<std::pair<LandmarkObs, LandmarkObs>>& dl,
	  const double s_x, const double s_y)
{
	assert(dl.size() > 0);

	double return_weight = 1;  // return weight for the particle
	for (const auto& p : dl)
	{
		const auto& landmark = p.first;
		const auto& observation = p.second;
			// calculate the weights
			double C = (1 / (2 * 3.14*s_x * s_y));
			return_weight = return_weight * (C*exp(-((landmark.x - observation.x)*(landmark.x - observation.x) / (2.0 * s_x * s_x)) +
			(landmark.y - observation.y)*(landmark.y - observation.y) / (2.0* s_y * s_y)));
			cout << "Data Associaion list first x and y  (" << landmark.x << " , " << landmark.y << ")  , (" << observation.x << " , " << observation.y << endl;
	}

	return return_weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
std::vector<LandmarkObs> observations, Map map_landmarks)
{
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
// For each particle find the landmarks less than sensor range 
	for (int i = 0; i < particles.size(); i++) {
		double dist_particle_to_lm = INFINITY;
		std::vector<LandmarkObs> in_range_lm_list;
		Particle particle = particles[i];
		double x_par = particles[i].x;
		double y_par = particles[i].y;
		double weight;
		// For each map landmark within sensor range to particle 
		// 1. Make list of all landmarks within sensor range of particle, call this `in_range_lm_list`
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			double x_map_lm = map_landmarks.landmark_list[j].x_f;
			double y_map_lm = map_landmarks.landmark_list[j].y_f;
			dist_particle_to_lm = dist(x_par, y_par, x_map_lm, y_map_lm);
			// if the landmark is less than sensor range then add it to the in_range_lm_list for the particle
			if (dist_particle_to_lm <= sensor_range) {
				// create an object in_range_lm_object to push LandmarkObs onto the in_range_lm_list
				LandmarkObs  in_range_lm_object;
				in_range_lm_object.id = map_landmarks.landmark_list[j].id_i; // original particle id
				in_range_lm_object.x = map_landmarks.landmark_list[j].x_f;  // landmark in range x value
				in_range_lm_object.y = map_landmarks.landmark_list[j].y_f;
				in_range_lm_list.push_back(in_range_lm_object);  // add the in range value to the list
				
			}   // end of if
		}  // end of inner for 
		//2. Convert all observations from local to global frame, call this `transformed_obs`

		vector<LandmarkObs> transformed_obs = STransformObservationToMapCoordinates(observations, particles[i]);

		//  3. Perform `dataAssociation`. This will put the index of the `in_range_lm_list` nearest to each 
		//   `transformed_obs` in the `id` field of the `transformed_obs`.
		// make a pair

		std::list<std::pair<LandmarkObs, LandmarkObs>> DataAssociationList = SDataAssociation(in_range_lm_list, transformed_obs);
		double pweight = WeightParticleFromNearestObservation(DataAssociationList,
				std_x, std_y);
		}  // end of outer for loop
	}
  
		

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled_particles;
	
	// put all particles.weight into the weights vector
	for (size_t i = 0; i < particles.size(); i++)
	{
		weights.push_back(particles[i].weight);
	}
	default_random_engine gen;
	// distribute the weights randomly from weights.begin to
	// weights. end using discrete_distribution function
	discrete_distribution<int> weight_distribution(weights.begin(), weights.end());

	

	for (int i = 0; i < particles.size(); i++) {
		int weighted_index = weight_distribution(gen);
		// what is range of weighted_index
		resampled_particles.push_back(particles[weighted_index]);
		
	}

	particles = resampled_particles;
	weights.clear();
	for (int i = 0; i<particles.size(); i++) {
		particles[i].weight = 1.;
	}
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
