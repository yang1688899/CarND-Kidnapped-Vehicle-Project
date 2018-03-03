/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include<tuple>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for (int i; i<num_particles; i++){
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
	}
	is_initialized = true; 

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	for (int i=0; i<particles.size(); i++){
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
		double noise_x = dist_x(gen);
		double noise_y = dist_y(gen);
		double noise_theta = dist_theta(gen);
		
		particles[i].x = noise_x + (velocity/yaw_rate)*(sin(noise_theta + yaw_rate*delta_t)-sin(noise_theta));
		particles[i].y = noise_y + (velocity/yaw_rate)*(cos(noise_theta)+cos(noise_theta+yaw_rate*delta_t));
		particles[i].theta = noise_theta + yaw_rate*delta_t;
	}


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

vector<LandmarkObs> ParticleFilter::transformation(Particle p, const vector<LandmarkObs> observations){
	vector <LandmarkObs> obs_transformed;
	for(int i=0; i<observations.size(); i++){
		LandmarkObs obs = observations[i];
		double x_map = p.x + cos(p.theta)*obs.x - sin(p.theta)*obs.y;
		double y_map = p.y + sin(p.theta)*obs.x + cos(p.theta)*obs.y;
		obs.x = x_map;
		obs.y = y_map;
		obs_transformed.push_back(obs);
	}
	return obs_transformed;
}

vector<Map::single_landmark_s> ParticleFilter::association(vector<LandmarkObs> obs_transformed,const Map &map_landmarks, double sensor_range){
	double min;
	vector<Map::single_landmark_s> associated_landmark_list;
	int flag;

	vector<Map::single_landmark_s> landmark_list = map_landmarks.landmark_list;
	for (int i; i<obs_transformed.size(); i++){
		min = sensor_range;
		for (int j=0; j<landmark_list.size(); j++){
			double distance = dist(obs_transformed[i].x, obs_transformed[i].y, landmark_list[j].x_f, landmark_list[j].y_f);
			if (distance<min){
				min = distance;
				flag = j;
			}
		}
		
		associated_landmark_list.push_back(landmark_list[flag]);
		landmark_list.erase(landmark_list.begin()+flag);
	}
	return associated_landmark_list;
}


double ParticleFilter::calculate_weight(vector<LandmarkObs> obs_transformed, vector<Map::single_landmark_s> associated_landmark_list, double std_landmark[]){
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	double weight = 1.0;
	for (int i=0; obs_transformed.size(); i++){
		double obs_x = obs_transformed[i].x;
		double obs_y = obs_transformed[i].y;
		double map_x = associated_landmark_list[i].x_f;
		double map_y = associated_landmark_list[i].y_f;
		double prob = multi_gauss_prob_den(obs_x, obs_y, map_x, map_y, sig_x, sig_y);
		weight *= prob;
	}
	return weight;
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	for (int i; i<particles.size(); i++){
		vector<LandmarkObs> obs_transformed = transformation(particles[i],observations);
		vector<Map::single_landmark_s> associated_landmark_list = association(obs_transformed, map_landmarks,sensor_range);
		particles[i].weight= calculate_weight(obs_transformed, associated_landmark_list, std_landmark);

	}

}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	vector<double> weights;
	vector<Particle> resample_list;
	for(int i=0; i<particles.size(); i++){
		weights.push_back(particles[i].weight);
	}
	discrete_distribution<> dist(weights.begin(),weights.end());
	for (int i=0; i<particles.size(); i++){
		int pos = dist(gen);
		resample_list.push_back(particles[pos]);
	}
	particles = resample_list;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
