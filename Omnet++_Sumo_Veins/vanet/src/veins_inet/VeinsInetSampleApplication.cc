//// with the portion
// Copyright (C) 2018 Christoph Sommer <sommer@ccs-labs.org>
//
// Documentation for these modules is at http://veins.car2x.org/
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#include "veins_inet/VeinsInetSampleApplication.h"

#include "inet/common/ModuleAccess.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/TagBase_m.h"
#include "inet/common/TimeTag_m.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/networklayer/common/L3AddressTag_m.h"
#include "inet/transportlayer/contract/udp/UdpControlInfo_m.h"

#include "veins_inet/VeinsInetSampleMessage_m.h"

#include "inet/common/geometry/common/Coord.h"

#include <fstream>

#include "veins/modules/mobility/traci/TraCIMobility.h"
#include <cstdlib>  // For system()
#include <string>   // For std::to_string
#include <set> // Add this for the std::set
#include <random> // For random node selection
#include <memory>  // For std::shared_ptr and std::make_shared

#include <cmath>     // sqrt
#include <cstring>   // memset, memcpy
#include <errno.h>   // errno

#include "inet/common/IntrusivePtr.h"

using namespace inet;

#pragma pack(push, 1)  // Ensure consistent packing
struct MessageData {
    char sender[100];
    char receiverId[100];
    double posx;
    double posy;
    double spdx;
    double spdy;
    double aclx;
    double acly;
    double hedx;
    double hedy;
    double sendTime;
    char attackType[50];
    int label;
};
#pragma pack(pop)

//____________________________________________ Static variables ________________________________ //
static std::set<std::tuple<int, int, double>> maliciousMessages; // Track malicious messages
static double maliciousMessageExpiryTime = 1.0; // Expire malicious entries after 3 seconds
//static std::vector<int> allowedNodes = {10, 12, 4, 9, 2, 6, 18, 3, 0, 1, 19, 5}; // normal message senders
//static std::vector<int> attackerNodes = {0, 3, 4, 5, 10, 18, 1, 15};

//static std::vector<int> attackerNodes = {16, 17, 4, 3, 9, 0, 15, 2, 1, 19, 11, 5};
//static std::vector<int> allowedNodes = {1, 5, 9, 15, 0, 17, 2, 7, 11, 13, 3, 8}; // eligible nodes to send messages
//static std::vector<int> allowedNodes = {0,17,16,2,9,5,19,11,4,3,15}; // eligible nodes to send messages
static std::map<std::pair<int, int>, simtime_t> lastMessageTimes; // Track last message time for each sender-receiver pair

// control attack types
//static std::string SINGLE_ATTACK_MODE = "DoS";
//static std::string SINGLE_ATTACK_MODE = "Replay";
static std::string SINGLE_ATTACK_MODE = "RandomSpeed";
//static std::string SINGLE_ATTACK_MODE = "RandomPosition";
//static std::string SINGLE_ATTACK_MODE = "ConstantPositionOffset";
//static std::string SINGLE_ATTACK_MODE = "ConstantSpeedOffset";
//static std::string SINGLE_ATTACK_MODE = "";   // mixed (original)


//_______ orlando ___________//
//static std::vector<int> allowedNodes = {40, 83, 60, 74, 15, 86, 84, 48, 33, 2, 61, 92, 96, 29, 80, 70, 9, 31, 14, 64, 89, 7, 12, 71, 32, 50, 81, 30, 78, 94, 73, 34, 72, 25, 35, 93, 22, 23, 63, 5, 36, 95, 69, 24, 19, 66, 27, 56, 51, 62}; // normal message senders
//static std::vector<int> attackerNodes = {9, 63, 23, 95, 12, 24, 96, 61, 40, 31, 25, 74, 94, 50, 73, 66, 72, 69, 32, 33, 34, 86, 48, 36, 2, 71, 92, 93, 78, 5, 29, 83, 84, 15, 70, 22, 62, 30, 14, 27, 56, 81, 51, 7, 64};


//_______ casa ___________//
//static std::vector<int> allowedNodes = {26, 16, 57, 17, 41, 11, 30, 3, 58, 31, 34, 33, 44, 38, 19, 47, 6, 10, 42, 0, 18, 35, 15, 56, 37, 24, 50, 1, 25, 55, 14, 21, 59, 12, 8, 46, 28, 2, 5, 27}; // normal message senders
//static std::vector<int> attackerNodes = {17, 30, 12, 31, 14, 18, 28, 33, 6, 24, 15, 0, 16, 56, 21, 37, 26, 58, 19, 38, 11, 25, 10, 57, 34, 59, 42, 47, 2, 1, 44, 55, 50, 8, 46};

//_______ origin 20 vehicles ___________//
//static std::vector<int> allowedNodes = {11, 14, 9, 8, 17, 16, 10, 0, 13, 5, 19, 12, 7, 4, 1, 6}; // normal message senders
//static std::vector<int> attackerNodes = {6, 13, 10, 9, 4, 11, 12, 0, 5, 8};


//_______________________________________ percentages ___________________//
//_____ 20 vehicles _____//
//static std::vector<int> allowedNodes = {4, 3, 12, 17, 10, 5, 15, 8, 1, 16, 11, 2, 6, 19, 7, 13, 14, 18, 9, 0};
//static std::vector<int> attackerNodes = {18, 19}; // 10%
//static std::vector<int> attackerNodes = {13, 14, 9, 2}; // 20%
//static std::vector<int> attackerNodes = {0, 16, 13, 15, 18, 6}; // 30%
//static std::vector<int> attackerNodes = {6, 3, 14, 19, 17, 0, 12, 7, 13, 4}; // 50%


//_______________________________________ percentages ___________________//
//_____ 50 vehicles _____//
static std::vector<int> allowedNodes = {16, 29, 0, 14, 27, 38, 15, 30, 41, 36, 19, 12, 46, 13, 2, 26, 49, 20, 45, 37, 28, 17, 18, 1, 21, 35, 9, 42, 25, 40, 32, 47, 6, 4, 3, 33, 23, 5, 10, 11, 8, 44, 24, 48, 43, 7, 22, 39, 34, 31};
static std::vector<int> attackerNodes = {17, 3, 18, 10, 6}; // 10%
//static std::vector<int> attackerNodes = {17, 3, 18, 10, 6, 9, 29, 22, 28, 14}; // 20%
//static std::vector<int> attackerNodes = {45, 19, 11, 3, 48, 21, 49, 44, 37, 43, 12, 0, 10, 46, 20}; // 30%
//static std::vector<int> attackerNodes = {17, 3, 18, 10, 6, 9, 29, 22, 28, 14, 23, 33, 1, 30, 25, 27, 39, 20, 38, 31, 42, 24, 17, 19, 8}; // 50%

//_______________________________________ percentages ___________________//
//_____ 100 vehicles _____//
//static std::vector<int> allowedNodes = {41, 89, 29, 35, 55, 6, 23, 0, 77, 1, 94, 34, 87, 10, 16, 84, 57, 53, 8, 68, 70, 62, 31, 37, 96, 3, 49, 26, 83, 38, 46, 28, 12, 39, 45, 27, 75, 64, 63, 88, 86, 99, 80, 13, 91, 42, 11, 30, 4, 50, 71, 44, 43, 24, 73, 92, 78, 15, 48, 72, 17, 20, 85, 18, 32, 47, 14, 33, 93, 9, 79, 76, 21, 51, 19, 67, 5, 60, 90, 25, 58, 7, 59, 82, 74, 66, 52, 69, 2, 95, 40, 98, 65, 61, 81, 56, 97, 54, 36, 22};
//static std::vector<int> attackerNodes = {2, 15, 20, 11, 23, 7, 40, 31, 18, 16}; // 10%
//static std::vector<int> attackerNodes = {20, 2, 15, 11, 22, 33, 10, 7, 40, 31, 20, 14, 30, 39, 24, 21, 43, 34, 32, 51}; // 20%
//static std::vector<int> attackerNodes = {1, 49, 40, 57, 2, 31, 41, 15, 25, 7, 55, 18, 10, 32, 54, 28, 43, 70, 23, 51, 10, 14, 59, 95, 50, 80, 34, 69, 11, 35}; // 30%
//static std::vector<int> attackerNodes = {12, 91, 49, 55, 47, 40, 77, 41, 66, 51, 16, 54, 28, 25, 2, 9, 33, 8, 82, 21, 44, 39, 30, 6, 45, 87, 86, 96, 60, 70, 11, 31, 93, 75, 95, 62, 48, 68, 29, 36, 46, 14, 43, 63, 53, 24, 23, 50, 81, 59}; // 50%


//_______________________________________ percentages ___________________//
//_____ 200 vehicles _____//
//static std::vector<int> allowedNodes = {18, 115, 192, 166, 67, 197, 184, 29, 110, 111, 11, 69, 32, 135, 97, 80, 128, 37, 120, 42, 91, 93, 139, 70, 66, 48, 100, 59, 172, 190, 146, 161, 22, 57, 140, 159, 174, 125, 123, 86, 126, 155, 101, 20, 157, 89, 136, 144, 64, 185, 106, 1, 58, 122, 54, 143, 129, 5, 56, 189, 17, 39, 162, 99, 164, 98, 116, 10, 163, 71, 171, 4, 175, 68, 15, 28, 196, 45, 40, 168, 165, 84, 31, 147, 90, 78, 16, 199, 187, 109, 142, 61, 81, 3, 186, 121, 118, 145, 83, 23, 152, 26, 132, 49, 94, 25, 193, 182, 79, 46, 138, 177, 180, 188, 88, 87, 133, 13, 178, 119, 82, 148, 112, 27, 53, 62, 55, 183, 50, 179, 137, 7, 160, 167, 41, 107, 124, 158, 51, 102, 103, 150, 151, 104, 43, 60, 0, 95, 8, 194, 108, 76, 19, 113, 44, 191, 195, 154, 131, 134, 130, 73, 47, 9, 75, 38, 198, 30, 105, 176, 24, 77, 149, 141, 96, 35, 14, 72, 52, 170, 156, 12, 92, 74, 33, 6, 173, 63, 114, 127, 21, 169, 34, 85, 2, 181, 153, 65, 117, 36};
//static std::vector<int> allowedNodes = {0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12, 13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38, 39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51, 52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64, 65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77, 78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90, 91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199};
//static std::vector<int> attackerNodes = {2, 164, 145, 52, 179, 1, 54, 132, 8, 51, 148, 43, 105, 133, 77, 174, 64, 126, 142, 11, 191, 18, 73, 123, 19, 7, 139, 138, 85, 52, 47, 40, 48, 190, 4, 30, 146, 57, 111, 40}; // 10%
//static std::vector<int> attackerNodes = {20, 2, 15, 11, 22, 33, 10, 7, 40, 31, 20, 14, 30, 39, 24, 21, 43, 34, 32, 51}; // 20%
//static std::vector<int> attackerNodes = {169, 192, 71, 154, 136, 22, 161, 96, 173, 32, 138, 63, 6, 185, 187, 126, 168, 44, 79, 109, 189, 151, 124, 106, 31, 143, 127, 8, 196, 125, 58, 153, 51, 80, 53, 157, 113, 170, 186, 172, 77, 190, 195, 115, 167, 166, 76, 98, 66, 152, 123, 26, 74, 142, 102, 88, 183, 17, 95, 90}; // 30%
//static std::vector<int> attackerNodes = {11, 1, 9, 66, 120, 97, 55, 10, 138, 152, 99, 21, 126, 49, 20, 26, 144, 54, 111, 30, 63, 127, 157, 196, 132, 29, 170, 133, 193, 8, 124, 35, 168, 44, 178, 52, 93, 0, 60, 70, 184, 104, 114, 159, 156, 27, 113, 130, 165, 67, 40, 173, 150, 101, 33, 50, 92, 181, 2, 6, 166, 16, 24, 162, 175, 57, 141, 163, 147, 183, 18, 136, 75, 58, 125, 154, 129, 198, 98, 187, 161, 192, 140, 0, 77, 72, 18, 85, 179, 122, 158, 59, 131, 91, 68, 48, 42, 67, 69, 14}; // 50%

Define_Module(VeinsInetSampleApplication);

//___________________________________ Helper function for random seed generation _____________________________//
uint32_t VeinsInetSampleApplication::getRandomSeed() {
    return static_cast<uint32_t>(
        simTime().raw() * 1000000 +
        getParentModule()->getId() +
        std::hash<std::string>{}(std::to_string(std::time(nullptr))) +
        std::hash<std::string>{}(getParentModule()->getFullName()) +
        static_cast<uint32_t>(std::time(nullptr))
    );
}

// ___________________________________ Constructor/Destructor ________________________________//
VeinsInetSampleApplication::VeinsInetSampleApplication() {}
VeinsInetSampleApplication::~VeinsInetSampleApplication() {}

//____________________________________ DoS Attack Implementation _____________________________//
void VeinsInetSampleApplication::simulateDoSDisruptive() {
    int myIndex = getParentModule()->getIndex();
    const int numAttackers = 10;
    static const double PATTERN_CHANGE_INTERVAL = 5.0;

    // Create attacker selection with new random seed
    std::vector<int> currentNodes = attackerNodes;
    std::mt19937 gen(getRandomSeed());
    std::shuffle(currentNodes.begin(), currentNodes.end(), gen);

    std::vector<int> currentAttackers(currentNodes.begin(),
        currentNodes.begin() + std::min(numAttackers, (int)currentNodes.size()));

    if (std::find(currentAttackers.begin(), currentAttackers.end(), myIndex) == currentAttackers.end()) {
        return;
    }

    // Create malicious message
    auto payload = makeShared<VeinsInetSampleMessage>();
    payload->setChunkLength(B(100));
    payload->setSenderId(getParentModule()->getFullName());
    payload->setMalicious(true);
    payload->setAttackType("DoS_Disruptive");

    // Generate malicious values
    auto position = mobility->getCurrentPosition();
    double timePhase = std::fmod(simTime().dbl(), PATTERN_CHANGE_INTERVAL) / PATTERN_CHANGE_INTERVAL;

    double baseSpeed = 30.0;
    double speedVariation = 20.0;
    double maliciousSpeedMagnitude = baseSpeed + speedVariation * std::sin(2 * M_PI * timePhase);

    double angle = 2 * M_PI * timePhase;
    double maliciousSpeedX = maliciousSpeedMagnitude * std::cos(angle);
    double maliciousSpeedY = maliciousSpeedMagnitude * std::sin(angle);

    // Set payload values
    payload->setPosx(position.x);
    payload->setPosy(position.y);
    payload->setSpdx(maliciousSpeedX);
    payload->setSpdy(maliciousSpeedY);

    double maxAccel = 8.0;
    payload->setAclx(maxAccel * std::cos(angle + M_PI/4));
    payload->setAcly(maxAccel * std::sin(angle + M_PI/4));

    // Set heading
    double magnitude = sqrt(maliciousSpeedX * maliciousSpeedX + maliciousSpeedY * maliciousSpeedY);
    if (magnitude > 0) {
        payload->setHedx(maliciousSpeedX / magnitude);
        payload->setHedy(maliciousSpeedY / magnitude);
    } else {
        payload->setHedx(0);
        payload->setHedy(0);
    }

    sendToNearbyNodes(payload, 6);
}

//____________________________________ Replay Attack Implementation _____________________________//
void VeinsInetSampleApplication::simulateReplayAttack() {
    int myIndex = getParentModule()->getIndex();
    const int numAttackers = 10;

    // Create attacker selection with new random seed
    std::vector<int> currentNodes = attackerNodes;
    std::mt19937 gen(getRandomSeed());
    std::shuffle(currentNodes.begin(), currentNodes.end(), gen);

    std::vector<int> currentAttackers(currentNodes.begin(),
        currentNodes.begin() + std::min(numAttackers, (int)currentNodes.size()));

    // Check if current node is an attacker
    if (std::find(currentAttackers.begin(), currentAttackers.end(), myIndex) == currentAttackers.end()) {
        return;
    }

    // Static storage for previously captured messages
    static std::vector<inet::IntrusivePtr<VeinsInetSampleMessage>> capturedMessages;

    // Capture a new message or replay an existing one
    if (capturedMessages.size() < 5) {  // Limit to storing 5 messages
        // Capture current state as a potential replay message
        auto currentPosition = mobility->getCurrentPosition();
        auto currentSpeed = mobility->getCurrentVelocity();

        // Create IntrusivePtr directly
        inet::IntrusivePtr<VeinsInetSampleMessage> payload = makeShared<VeinsInetSampleMessage>();
        payload->setChunkLength(B(100));
        payload->setSenderId(getParentModule()->getFullName());
        payload->setMalicious(true);
        payload->setAttackType("ReplayAttack");

        // Calculate acceleration with smoothing
        double aclx = 0.0;
        double acly = 0.0;

        static std::map<int, Coord> prevSpeeds;
        static std::map<int, simtime_t> prevTimes;

        simtime_t currentTime = simTime();

        if (prevSpeeds.find(myIndex) != prevSpeeds.end()) {
            double dt = (currentTime - prevTimes[myIndex]).dbl();
            if (dt > 0) {
                // Calculate instantaneous acceleration
                double inst_aclx = (currentSpeed.x - prevSpeeds[myIndex].x) / dt;
                double inst_acly = (currentSpeed.y - prevSpeeds[myIndex].y) / dt;

                // Apply smoothing
                const double alpha = 0.3;
                aclx = alpha * inst_aclx + (1 - alpha) * prevSpeeds[myIndex].x;
                acly = alpha * inst_acly + (1 - alpha) * prevSpeeds[myIndex].y;

                // Limit acceleration
                const double MAX_ACCEL = 5.0;
                aclx = std::max(std::min(aclx, MAX_ACCEL), -MAX_ACCEL);
                acly = std::max(std::min(acly, MAX_ACCEL), -MAX_ACCEL);
            }
        }

        // Update state tracking
        prevSpeeds[myIndex] = currentSpeed;
        prevTimes[myIndex] = currentTime;

        // Calculate heading
        double speed_magnitude = std::sqrt(currentSpeed.x * currentSpeed.x + currentSpeed.y * currentSpeed.y);
        double headx = 0.0;
        double heady = 0.0;

        if (speed_magnitude > 0.1) {
            headx = currentSpeed.x / speed_magnitude;
            heady = currentSpeed.y / speed_magnitude;
        } else if (prevSpeeds.find(myIndex) != prevSpeeds.end()) {
            double prev_magnitude = std::sqrt(
                prevSpeeds[myIndex].x * prevSpeeds[myIndex].x +
                prevSpeeds[myIndex].y * prevSpeeds[myIndex].y
            );
            if (prev_magnitude > 0.1) {
                headx = prevSpeeds[myIndex].x / prev_magnitude;
                heady = prevSpeeds[myIndex].y / prev_magnitude;
            }
        }

        // Set payload values
        payload->setPosx(currentPosition.x);
        payload->setPosy(currentPosition.y);
        payload->setSpdx(currentSpeed.x);
        payload->setSpdy(currentSpeed.y);
        payload->setAclx(aclx);
        payload->setAcly(acly);
        payload->setHedx(headx);
        payload->setHedy(heady);

        // Store the captured message
        capturedMessages.push_back(payload);

        EV_INFO << "Replay Attack: Captured Message from Node " << myIndex << endl;
    }
    else {
        // Replay a previously captured message
        // Randomly select a message to replay
        std::uniform_int_distribution<> dist(0, capturedMessages.size() - 1);
        int replayIndex = dist(gen);

        // Create a copy of the captured message
        inet::IntrusivePtr<VeinsInetSampleMessage> replayPayload = makeShared<VeinsInetSampleMessage>(*capturedMessages[replayIndex]);

        // Modify the payload to indicate it's a replay
        replayPayload->setAttackType("Replay_Attack");

        EV_INFO << "Replay Attack: Replaying Captured Message from Node " << myIndex
                << " (Original Sender: " << replayPayload->getSenderId() << ")" << endl;

        // Send the replayed message
        sendToNearbyNodes(replayPayload, 4);  // Assuming 50% malicious ratio
    }
}


//_______________________________ Random Speed Attack Implementation ___________________________//
void VeinsInetSampleApplication::simulateRandomSpeed() {
    int myIndex = getParentModule()->getIndex();
    const int numAttackers = 12;

    // Select attackers
    std::vector<int> currentNodes = attackerNodes;
    std::mt19937 gen(getRandomSeed());
    std::shuffle(currentNodes.begin(), currentNodes.end(), gen);

    std::vector<int> currentAttackers(currentNodes.begin(),
        currentNodes.begin() + std::min(numAttackers, (int)currentNodes.size()));

    if (std::find(currentAttackers.begin(), currentAttackers.end(), myIndex) == currentAttackers.end()) {
        return;
    }

    auto payload = makeShared<VeinsInetSampleMessage>();
    payload->setChunkLength(B(100));
    payload->setSenderId(getParentModule()->getFullName());
    payload->setMalicious(true);
    payload->setAttackType("Random_speed");

    auto position = mobility->getCurrentPosition();
    auto speed = mobility->getCurrentVelocity();

    // Generate random speed variations
    std::normal_distribution<> speedDist(0, 15.0/3);
    double maliciousSpeedX = speed.x + speedDist(gen);
    double maliciousSpeedY = speed.y + speedDist(gen);

    payload->setPosx(position.x);
    payload->setPosy(position.y);
    payload->setSpdx(maliciousSpeedX);
    payload->setSpdy(maliciousSpeedY);

    // Generate random acceleration
    std::normal_distribution<> accelDist(0, 1.0);
    payload->setAclx(accelDist(gen));
    payload->setAcly(accelDist(gen));

    // Set heading
    double magnitude = sqrt(maliciousSpeedX * maliciousSpeedX + maliciousSpeedY * maliciousSpeedY);
    if (magnitude > 0) {
        payload->setHedx(maliciousSpeedX / magnitude);
        payload->setHedy(maliciousSpeedY / magnitude);
    } else {
        payload->setHedx(0);
        payload->setHedy(0);
    }

    sendToNearbyNodes(payload, 4);
}

//_______________________________ Random position Attack Implementation ___________________________//
void VeinsInetSampleApplication::simulateRandomPosition() {
    int myIndex = getParentModule()->getIndex();
    const int numAttackers = 10;

    // Create attacker selection with new random seed
    std::vector<int> currentNodes = attackerNodes;
    std::mt19937 gen(getRandomSeed());
    std::shuffle(currentNodes.begin(), currentNodes.end(), gen);

    std::vector<int> currentAttackers(currentNodes.begin(),
        currentNodes.begin() + std::min(numAttackers, (int)currentNodes.size()));

    // Check if current node is an attacker
    if (std::find(currentAttackers.begin(), currentAttackers.end(), myIndex) == currentAttackers.end()) {
        return;
    }

    // Get current state
    auto currentPosition = mobility->getCurrentPosition();
    auto currentSpeed = mobility->getCurrentVelocity();

    // Calculate acceleration with smoothing
    double aclx = 0.0;
    double acly = 0.0;

    static std::map<int, Coord> prevSpeeds;
    static std::map<int, simtime_t> prevTimes;

    simtime_t currentTime = simTime();

    if (prevSpeeds.find(myIndex) != prevSpeeds.end()) {
        double dt = (currentTime - prevTimes[myIndex]).dbl();
        if (dt > 0) {
            // Calculate instantaneous acceleration
            double inst_aclx = (currentSpeed.x - prevSpeeds[myIndex].x) / dt;
            double inst_acly = (currentSpeed.y - prevSpeeds[myIndex].y) / dt;

            // Apply smoothing
            const double alpha = 0.3;
            aclx = alpha * inst_aclx + (1 - alpha) * prevSpeeds[myIndex].x;
            acly = alpha * inst_acly + (1 - alpha) * prevSpeeds[myIndex].y;

            // Limit acceleration
            const double MAX_ACCEL = 5.0;
            aclx = std::max(std::min(aclx, MAX_ACCEL), -MAX_ACCEL);
            acly = std::max(std::min(acly, MAX_ACCEL), -MAX_ACCEL);
        }
    }

    // Update state tracking
    prevSpeeds[myIndex] = currentSpeed;
    prevTimes[myIndex] = currentTime;

    // Generate random position offset
    std::uniform_real_distribution<> dist_x(-100.0, 100.0);  // Random X offset between -100 and 100
    std::uniform_real_distribution<> dist_y(-100.0, 100.0);  // Random Y offset between -100 and 100

    double randomOffsetX = dist_x(gen);
    double randomOffsetY = dist_y(gen);

    // Calculate malicious position
    double maliciousPosx = currentPosition.x + randomOffsetX;
    double maliciousPosy = currentPosition.y + randomOffsetY;

    // Calculate heading (based on current speed or previous speed)
    double speed_magnitude = std::sqrt(currentSpeed.x * currentSpeed.x + currentSpeed.y * currentSpeed.y);
    double headx = 0.0;
    double heady = 0.0;

    if (speed_magnitude > 0.1) {
        headx = currentSpeed.x / speed_magnitude;
        heady = currentSpeed.y / speed_magnitude;
    } else if (prevSpeeds.find(myIndex) != prevSpeeds.end()) {
        double prev_magnitude = std::sqrt(
            prevSpeeds[myIndex].x * prevSpeeds[myIndex].x +
            prevSpeeds[myIndex].y * prevSpeeds[myIndex].y
        );
        if (prev_magnitude > 0.1) {
            headx = prevSpeeds[myIndex].x / prev_magnitude;
            heady = prevSpeeds[myIndex].y / prev_magnitude;
        }
    }

    // Create malicious message
    auto payload = makeShared<VeinsInetSampleMessage>();
    payload->setChunkLength(B(100));
    payload->setSenderId(getParentModule()->getFullName());
    payload->setMalicious(true);
    payload->setAttackType("Random_Position");

    // Set payload values
    payload->setPosx(maliciousPosx);
    payload->setPosy(maliciousPosy);
    payload->setSpdx(currentSpeed.x);
    payload->setSpdy(currentSpeed.y);
    payload->setAclx(aclx);
    payload->setAcly(acly);
    payload->setHedx(headx);
    payload->setHedy(heady);

    // Optional: Add logging
    EV_INFO << "Random Position Attack: "
            << "Node " << myIndex
            << " Original Position: (" << currentPosition.x << ", " << currentPosition.y << ")"
            << " Malicious Position: (" << maliciousPosx << ", " << maliciousPosy << ")"
            << " Offset: (" << randomOffsetX << ", " << randomOffsetY << ")"
            << endl;

    // Send to nearby nodes
    sendToNearbyNodes(payload, 4);  // Assuming 50% malicious ratio
}


//_____________________________________ Constant Position Offset Attack Implementation ____________________________//
void VeinsInetSampleApplication::simulateConstantPositionOffset() {
    int myIndex = getParentModule()->getIndex();
    const int numAttackers = 12;
    static std::map<int, Coord> positionOffsets;

    // Select attackers with new random seed
    std::vector<int> currentNodes = attackerNodes;
    std::mt19937 gen(getRandomSeed());
    std::shuffle(currentNodes.begin(), currentNodes.end(), gen);

    std::vector<int> currentAttackers(currentNodes.begin(),
        currentNodes.begin() + std::min(numAttackers, (int)currentNodes.size()));

    // Generate new offsets for current attackers
    std::normal_distribution<> offsetDist(0, 40.0);
    for (int attacker : currentAttackers) {
        if (positionOffsets.find(attacker) == positionOffsets.end()) {
            positionOffsets[attacker] = Coord(offsetDist(gen), offsetDist(gen), 0);
        }
    }

    if (std::find(currentAttackers.begin(), currentAttackers.end(), myIndex) == currentAttackers.end()) {
        return;
    }

    auto payload = makeShared<VeinsInetSampleMessage>();
    payload->setChunkLength(B(100));
    payload->setSenderId(getParentModule()->getFullName());
    payload->setMalicious(true);
    payload->setAttackType("Constant_Position_Offset");

    auto position = mobility->getCurrentPosition();
    auto speed = mobility->getCurrentVelocity();
    Coord offset = positionOffsets[myIndex];

    // Set modified position
    payload->setPosx(position.x + offset.x);
    payload->setPosy(position.y + offset.y);
    payload->setSpdx(speed.x);
    payload->setSpdy(speed.y);

    // Calculate acceleration
    double aclx = (speed.x - prevSpdx) / timeStep;
    double acly = (speed.y - prevSpdy) / timeStep;
    payload->setAclx(aclx);
    payload->setAcly(acly);

    // Update previous speeds
    prevSpdx = speed.x;
    prevSpdy = speed.y;

    // Set heading
    double magnitude = sqrt(speed.x * speed.x + speed.y * speed.y);
    if (magnitude > 0) {
        payload->setHedx(speed.x / magnitude);
        payload->setHedy(speed.y / magnitude);
    } else {
        payload->setHedx(0);
        payload->setHedy(0);
    }

    sendToNearbyNodes(payload, 4);
}

//_____________________________________ Constant Speed Offset Attack Implementation ____________________________//
void VeinsInetSampleApplication::simulateConstantSpeedOffset() {
    int myIndex = getParentModule()->getIndex();
    const int numAttackers = 10;
    const double CONSTANT_SPEED_OFFSET = 10.0;  // Constant speed offset in m/s

    // Create attacker selection with new random seed
    std::vector<int> currentNodes = attackerNodes;
    std::mt19937 gen(getRandomSeed());
    std::shuffle(currentNodes.begin(), currentNodes.end(), gen);

    std::vector<int> currentAttackers(currentNodes.begin(),
        currentNodes.begin() + std::min(numAttackers, (int)currentNodes.size()));

    // Check if current node is an attacker
    if (std::find(currentAttackers.begin(), currentAttackers.end(), myIndex) == currentAttackers.end()) {
        return;
    }

    // Get current state
    auto position = mobility->getCurrentPosition();
    auto speed = mobility->getCurrentVelocity();

    // Calculate acceleration with smoothing
    double aclx = 0.0;
    double acly = 0.0;

    static std::map<int, Coord> prevSpeeds;
    static std::map<int, simtime_t> prevTimes;

    simtime_t currentTime = simTime();

    if (prevSpeeds.find(myIndex) != prevSpeeds.end()) {
        double dt = (currentTime - prevTimes[myIndex]).dbl();
        if (dt > 0) {
            // Calculate instantaneous acceleration
            double inst_aclx = (speed.x - prevSpeeds[myIndex].x) / dt;
            double inst_acly = (speed.y - prevSpeeds[myIndex].y) / dt;

            // Apply smoothing
            const double alpha = 0.3;
            aclx = alpha * inst_aclx + (1 - alpha) * prevSpeeds[myIndex].x;
            acly = alpha * inst_acly + (1 - alpha) * prevSpeeds[myIndex].y;

            // Limit acceleration
            const double MAX_ACCEL = 5.0;
            aclx = std::max(std::min(aclx, MAX_ACCEL), -MAX_ACCEL);
            acly = std::max(std::min(acly, MAX_ACCEL), -MAX_ACCEL);
        }
    }

    // Update state tracking
    prevSpeeds[myIndex] = speed;
    prevTimes[myIndex] = currentTime;

    // Calculate speed with constant offset
    double currentSpeedMagnitude = std::sqrt(speed.x * speed.x + speed.y * speed.y);
    double currentSpeedAngle = std::atan2(speed.y, speed.x);

    // Add constant speed offset
    double maliciousSpeedMagnitude = currentSpeedMagnitude + CONSTANT_SPEED_OFFSET;

    // Calculate new speed components
    double maliciousSpeedX = maliciousSpeedMagnitude * std::cos(currentSpeedAngle);
    double maliciousSpeedY = maliciousSpeedMagnitude * std::sin(currentSpeedAngle);

    // Calculate heading
    double speed_magnitude = std::sqrt(maliciousSpeedX * maliciousSpeedX + maliciousSpeedY * maliciousSpeedY);
    double headx = 0.0;
    double heady = 0.0;

    if (speed_magnitude > 0.1) {
        headx = maliciousSpeedX / speed_magnitude;
        heady = maliciousSpeedY / speed_magnitude;
    } else if (prevSpeeds.find(myIndex) != prevSpeeds.end()) {
        double prev_magnitude = std::sqrt(
            prevSpeeds[myIndex].x * prevSpeeds[myIndex].x +
            prevSpeeds[myIndex].y * prevSpeeds[myIndex].y
        );
        if (prev_magnitude > 0.1) {
            headx = prevSpeeds[myIndex].x / prev_magnitude;
            heady = prevSpeeds[myIndex].y / prev_magnitude;
        }
    }

    // Create malicious message
    auto payload = makeShared<VeinsInetSampleMessage>();
    payload->setChunkLength(B(100));
    payload->setSenderId(getParentModule()->getFullName());
    payload->setMalicious(true);
    payload->setAttackType("Constant_Speed_Offset");

    // Set payload values
    payload->setPosx(position.x);
    payload->setPosy(position.y);
    payload->setSpdx(maliciousSpeedX);
    payload->setSpdy(maliciousSpeedY);
    payload->setAclx(aclx);
    payload->setAcly(acly);
    payload->setHedx(headx);
    payload->setHedy(heady);

    // Optional: Add logging
    EV_INFO << "Constant Speed Offset Attack: "
            << "Node " << myIndex
            << " Original Speed: (" << speed.x << ", " << speed.y << ") m/s"
            << " Malicious Speed: (" << maliciousSpeedX << ", " << maliciousSpeedY << ") m/s"
            << " Acceleration: (" << aclx << ", " << acly << ") m/s²"
            << " Heading: (" << headx << ", " << heady << ")"
            << endl;

    // Send to nearby nodes
    sendToNearbyNodes(payload, 4);  // Assuming 50% malicious ratio
}

//__________________________Nearby Nodes Message Sending Implementation _____________________________//
void VeinsInetSampleApplication::sendToNearbyNodes(const inet::IntrusivePtr<VeinsInetSampleMessage>& payload, int numReceivers) {
    auto systemModule = getSimulation()->getSystemModule();
    int myIndex = getParentModule()->getIndex();
    auto position = mobility->getCurrentPosition();

    // Find valid receivers
    std::vector<cModule*> nearbyNodes;
    for (cModule::SubmoduleIterator it(systemModule); !it.end(); ++it) {
        cModule* receiverModule = *it;
        int receiverIndex = receiverModule->getIndex();

        if (receiverIndex == myIndex) continue;

        auto mobilityModule = receiverModule->getSubmodule("mobility");
        if (!mobilityModule) continue;

        auto receiverMobility = check_and_cast<veins::VeinsInetMobility*>(mobilityModule);
        double distance = sqrt(pow(position.x - receiverMobility->getCurrentPosition().x, 2) +
                             pow(position.y - receiverMobility->getCurrentPosition().y, 2));
        if (distance <= distanceThreshold) {
            nearbyNodes.push_back(receiverModule);
        }
    }

    // Randomly select and send to receivers
    // Use time-based seed for randomization
    std::mt19937 g(static_cast<unsigned int>(simTime().raw() * myIndex + nearbyNodes.size()));
    std::shuffle(nearbyNodes.begin(), nearbyNodes.end(), g);

    int actualReceivers = std::min(static_cast<int>(nearbyNodes.size()), numReceivers);
    for (int i = 0; i < actualReceivers; ++i) {
        auto receiverModule = nearbyNodes[i];
        auto l3Address = L3AddressResolver().resolve(receiverModule->getFullPath().c_str(), L3AddressResolver::ADDR_IPv4);

        auto packet = createPacket("malicious");
        packet->insertAtBack(payload->dupShared());
        packet->addTag<L3AddressReq>()->setDestAddress(l3Address);

        sendPacket(std::move(packet));

        // Log the malicious message
        int receiverIndex = receiverModule->getIndex();
        maliciousMessages.insert(std::make_tuple(myIndex, receiverIndex, simTime().dbl()));
    }
}

//______________________________________ Application Start Implementation________________________________//
bool VeinsInetSampleApplication::startApplication() {
    // Track messages sent with a precise validity time
    static std::map<std::pair<int, int>, simtime_t> sentMessages; // {sender, receiver} -> last sent time
    static bool attackTriggered = false; // Flag to track attack start

    static std::map<int, Coord> prevPositions;
    static std::map<int, Coord> prevSpeeds;
    static std::map<int, simtime_t> prevTimes;

    // Normal message callback
    auto normalMessageCallback = [this]() {
        int myIndex = getParentModule()->getIndex();

        // Get current scenario configuration
        ScenarioConfig config = getScenarioConfig();

        // Create randomized sender selection
        std::vector<int> currentNodes = allowedNodes;
        std::mt19937 gen(getRandomSeed());
        std::shuffle(currentNodes.begin(), currentNodes.end(), gen);

        if (std::find(currentNodes.begin(), currentNodes.end(), myIndex) == currentNodes.end()) {
            return;
        }

        auto systemModule = getSimulation()->getSystemModule();
        simtime_t currentTime = simTime();

        // Clean expired messages
        for (auto it = maliciousMessages.begin(); it != maliciousMessages.end();) {
            if (currentTime - std::get<2>(*it) > SimTime(maliciousMessageExpiryTime, SIMTIME_S)) {
                it = maliciousMessages.erase(it);
            } else {
                ++it;
            }
        }

        // Get current state
        auto position = mobility->getCurrentPosition();
        auto speed = mobility->getCurrentVelocity();

        // Calculate acceleration with smoothing
        double aclx = 0.0;
        double acly = 0.0;

        if (prevSpeeds.find(myIndex) != prevSpeeds.end()) {
            double dt = (currentTime - prevTimes[myIndex]).dbl();
            if (dt > 0) {
                // Calculate instantaneous acceleration
                double inst_aclx = (speed.x - prevSpeeds[myIndex].x) / dt;
                double inst_acly = (speed.y - prevSpeeds[myIndex].y) / dt;

                // Apply smoothing
                const double alpha = 0.3;
                aclx = alpha * inst_aclx + (1 - alpha) * prevSpeeds[myIndex].x;
                acly = alpha * inst_acly + (1 - alpha) * prevSpeeds[myIndex].y;

                // Limit acceleration
                const double MAX_ACCEL = 5.0;
                aclx = std::max(std::min(aclx, MAX_ACCEL), -MAX_ACCEL);
                acly = std::max(std::min(acly, MAX_ACCEL), -MAX_ACCEL);
            }
        }

        // Update state tracking
        prevPositions[myIndex] = position;
        prevSpeeds[myIndex] = speed;
        prevTimes[myIndex] = currentTime;

        // Process receivers
        std::vector<cModule*> potentialReceivers;
        for (cModule::SubmoduleIterator it(systemModule); !it.end(); ++it) {
            cModule* receiverModule = *it;
            int receiverIndex = receiverModule->getIndex();

            // Skip invalid receivers
            if (receiverIndex == myIndex) continue;

            // Skip recent malicious message pairs
            if (std::any_of(maliciousMessages.begin(), maliciousMessages.end(),
                [myIndex, receiverIndex](const std::tuple<int, int, double>& entry) {
                    return std::get<0>(entry) == myIndex &&
                           std::get<1>(entry) == receiverIndex;
                })) {
                continue;
            }

            auto mobilityModule = receiverModule->getSubmodule("mobility");
            if (!mobilityModule) continue;

            // Check distance
            auto receiverMobility = check_and_cast<veins::VeinsInetMobility*>(mobilityModule);
            auto receiverPosition = receiverMobility->getCurrentPosition();
            double distance = sqrt(pow(position.x - receiverPosition.x, 2) +
                                 pow(position.y - receiverPosition.y, 2));

            if (distance <= distanceThreshold) {
                potentialReceivers.push_back(receiverModule);
            }
        }

        // Randomize receiver order
        std::shuffle(potentialReceivers.begin(), potentialReceivers.end(), gen);

        // Process each receiver
        for (auto receiverModule : potentialReceivers) {
            int receiverIndex = receiverModule->getIndex();

            // Check message deduplication
            auto messageKey = std::make_pair(myIndex, receiverIndex);
            if (sentMessages.find(messageKey) != sentMessages.end() &&
                currentTime - sentMessages[messageKey] < SimTime(3.0, SIMTIME_S)) {
                continue;
            }

            // Update sent time
            sentMessages[messageKey] = currentTime;

            // Create and send message
            auto payload = makeShared<VeinsInetSampleMessage>();
            payload->setChunkLength(B(100));
            payload->setSenderId(getParentModule()->getFullName());

            // Set position and speed
            payload->setPosx(position.x);
            payload->setPosy(position.y);
            payload->setSpdx(speed.x);
            payload->setSpdy(speed.y);
            payload->setAclx(aclx);
            payload->setAcly(acly);

            // Calculate heading
            double speed_magnitude = sqrt(speed.x * speed.x + speed.y * speed.y);
            double headx = 0.0;
            double heady = 0.0;

            if (speed_magnitude > 0.1) {
                headx = speed.x / speed_magnitude;
                heady = speed.y / speed_magnitude;
            } else if (prevSpeeds.find(myIndex) != prevSpeeds.end()) {
                double prev_magnitude = sqrt(
                    prevSpeeds[myIndex].x * prevSpeeds[myIndex].x +
                    prevSpeeds[myIndex].y * prevSpeeds[myIndex].y
                );
                if (prev_magnitude > 0.1) {
                    headx = prevSpeeds[myIndex].x / prev_magnitude;
                    heady = prevSpeeds[myIndex].y / prev_magnitude;
                }
            }

            payload->setHedx(headx);
            payload->setHedy(heady);

            // Send packet
            auto packet = createPacket("normal");
            packet->insertAtBack(payload);
            auto l3Address = L3AddressResolver().resolve(receiverModule->getFullPath().c_str());
            packet->addTag<L3AddressReq>()->setDestAddress(l3Address);

            sendPacket(std::move(packet));
        }
    };

    // Schedule normal messages
    //timerManager.create(veins::TimerSpecification(normalMessageCallback).interval(SimTime(timeStep, SIMTIME_S)));

    // Multiple Random Speed Attack windows
    //timerManager.create(veins::TimerSpecification([this]() {simulateRandomSpeed();}).oneshotIn(SimTime(60, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateRandomSpeed();}).oneshotIn(SimTime(40, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateRandomSpeed();}).oneshotIn(SimTime(65, SIMTIME_S)));

    //timerManager.create(veins::TimerSpecification([this]() {simulateRandomSpeed();}).interval(SimTime(5, SIMTIME_S)));

    //timerManager.create(veins::TimerSpecification([this]() {simulateDoSDisruptive();}).oneshotIn(SimTime(30, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateDoSDisruptive();}).oneshotIn(SimTime(45, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateDoSDisruptive();}).oneshotIn(SimTime(70, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateDoSDisruptive();}).oneshotIn(SimTime(90, SIMTIME_S)));

    //timerManager.create(veins::TimerSpecification([this]() {simulateDoSDisruptive();}).interval(SimTime(5, SIMTIME_S)));

    //timerManager.create(veins::TimerSpecification([this]() {simulateConstantPositionOffset();}).oneshotIn(SimTime(35, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateConstantPositionOffset();}).oneshotIn(SimTime(55, SIMTIME_S)));
    //timerManager.create(veins::TimerSpecification([this]() {simulateConstantPositionOffset();}).oneshotIn(SimTime(80, SIMTIME_S)));

    //timerManager.create(veins::TimerSpecification([this]() {simulateConstantPositionOffset();}).interval(SimTime(5, SIMTIME_S)));

    //_______________________________

    // Schedule messages with randomized timing
    //double baseOffset = static_cast<double>(getParentModule()->getIndex() % 5);

    // Normal messages
    //timerManager.create(veins::TimerSpecification(normalMessageCallback).interval(SimTime(timeStep, SIMTIME_S)));

    // Attack messages with varying intervals
    //timerManager.create(veins::TimerSpecification([this]() {if (simTime() >= 10) { double randomChance = static_cast<double>(rand()) / RAND_MAX; if (randomChance < 0.5) {  simulateDoSDisruptive();}}}).interval(SimTime(3.0, SIMTIME_S)));

    // Random Speed Attack with different timing
    //timerManager.create(veins::TimerSpecification([this]() { if (simTime() >= 15) { double randomChance = static_cast<double>(rand()) / RAND_MAX; if (randomChance < 0.7) { simulateRandomSpeed();}}}).interval(SimTime(4.0, SIMTIME_S)));

    // Position Offset Attack with its own timing
    //timerManager.create(veins::TimerSpecification([this]() {if (simTime() >= 20) { double randomChance = static_cast<double>(rand()) / RAND_MAX; if (randomChance < 0.6) { simulateConstantPositionOffset();}}}).interval(SimTime(5.0, SIMTIME_S)));

    //____________________________________

    // Schedule attack windows with clear time slots
    //timerManager.create(veins::TimerSpecification([this]() {
        // Time windows for different attacks
      //  double currentTime = simTime().dbl();

        //if (currentTime >= 10 && currentTime <= 20) {
            // DoS attack window
          //  simulateDoSDisruptive();  // 60% malicious messages
        //}
        //else if (currentTime > 30 && currentTime <= 45) {
            // Random Speed attack window
          //  simulateRandomSpeed();    // 40% malicious messages
        //}
        //else if (currentTime > 50 && currentTime <= 65) {
            // Position Offset attack window
          //  simulateConstantPositionOffset();  // 30% malicious messages
        //}
    //}).interval(SimTime(3.0, SIMTIME_S)));  // Check every 3 seconds

    //________________________________________________

    // Schedule regular messages with scenario awareness
    timerManager.create(veins::TimerSpecification([this, normalMessageCallback]() {
        rotateScenario();
        ScenarioConfig config = getScenarioConfig();

        // Determine message type based on probabilities
        double random = static_cast<double>(rand()) / RAND_MAX;

        double dosProb = config.dosProb;
        double randomSpeedProb = config.randomSpeedProb;
        double positionOffsetProb = config.positionProb;
        double constantSpeedOffsetProb = config.constantSpeedOffsetProb;
        double replayAttackProb = config.replayAttackProb;
        double randomPositionProb = config.randomPositionProb;

        double totalAttackProb = dosProb + randomSpeedProb + positionOffsetProb +
                                  constantSpeedOffsetProb + randomPositionProb + replayAttackProb;


        if (simTime() >= 10) {
                    if (!SINGLE_ATTACK_MODE.empty()) {
                        //  Single attack mode
                        double attackChance = static_cast<double>(rand()) / RAND_MAX;
                        if (attackChance < totalAttackProb) {
                            if (SINGLE_ATTACK_MODE == "DoS") {
                                simulateDoSDisruptive();
                            } else if (SINGLE_ATTACK_MODE == "Replay") {
                                simulateReplayAttack();
                            } else if (SINGLE_ATTACK_MODE == "RandomSpeed") {
                                simulateRandomSpeed();
                            } else if (SINGLE_ATTACK_MODE == "RandomPosition") {
                                simulateRandomPosition();
                            } else if (SINGLE_ATTACK_MODE == "ConstantPositionOffset") {
                                simulateConstantPositionOffset();
                            } else if (SINGLE_ATTACK_MODE == "ConstantSpeedOffset") {
                                simulateConstantSpeedOffset();
                            }
                        } else {
                            normalMessageCallback();
                        }
                    } else {
                        //  Mixed mode: original behavior
                        if (random < dosProb) {
                            simulateDoSDisruptive();
                        }
                        else if (random < (dosProb + randomSpeedProb)) {
                            simulateRandomSpeed();
                        }
                        else if (random < (dosProb + randomSpeedProb + positionOffsetProb)) {
                            simulateConstantPositionOffset();
                        }
                        else if (random < (dosProb + randomSpeedProb + positionOffsetProb + constantSpeedOffsetProb)) {
                            simulateConstantSpeedOffset();
                        }
                        else if (random < (dosProb + randomSpeedProb + positionOffsetProb +
                                           constantSpeedOffsetProb + randomPositionProb)) {
                            simulateRandomPosition();
                        }
                        else if (random < (dosProb + randomSpeedProb + positionOffsetProb +
                                           constantSpeedOffsetProb + randomPositionProb + replayAttackProb)) {
                            simulateReplayAttack();
                        }
                        else {
                            normalMessageCallback();
                        }
                    }
                } else {
                    normalMessageCallback();
                }
            }).interval(SimTime(timeStep, SIMTIME_S)));

        //if (simTime() >= 10) {  // Start attacks after 10s
         //   if (random < dosProb) {
          //      simulateDoSDisruptive();
          //  }
          //  else if (random < (dosProb + randomSpeedProb)) {
            //    simulateRandomSpeed();
            //}
            //else if (random < (dosProb + randomSpeedProb + positionOffsetProb)) {
              //  simulateConstantPositionOffset();
            //}
            //else if (random < (dosProb + randomSpeedProb + positionOffsetProb + constantSpeedOffsetProb)) {
              //  simulateConstantSpeedOffset();
            //}
            //else if (random < (dosProb + randomSpeedProb + positionOffsetProb +
              //                 constantSpeedOffsetProb + randomPositionProb)) {
                //simulateRandomPosition();
            //}
            //else if (random < (dosProb + randomSpeedProb + positionOffsetProb +
              //                 constantSpeedOffsetProb + randomPositionProb + replayAttackProb)) {
                //simulateReplayAttack();
            //}
            //else {
              //  normalMessageCallback();
            //}
        //} else {
          //  normalMessageCallback();
        //}
    //}).interval(SimTime(timeStep, SIMTIME_S)));


    return true;
}


VeinsInetSampleApplication::ScenarioConfig VeinsInetSampleApplication::getScenarioConfig() {
    ScenarioConfig config;
    switch(currentScenario) {
        case SPARSE_TRAFFIC:
            config.dosProb = 0.10;         // Reduced DoS probability 0.15
            config.replayAttackProb = 0.10;
            config.randomSpeedProb = 0.10;  // Reduced Random Speed 0.15
            config.randomPositionProb = 0.05;
            config.positionProb = 0.25;     // Increased Position Offset 0.30
            config.constantSpeedOffsetProb = 0.20;
            config.messageInterval = 3;
            config.numAttackers = 10;       // Increased attackers
            break;

        case DENSE_TRAFFIC:
            config.dosProb = 0.15; // 0.20
            config.replayAttackProb = 0.15;
            config.randomSpeedProb = 0.10; //0.15
            config.randomPositionProb = 0.10;
            config.positionProb = 0.20;     // Maintained high 0.25
            config.constantSpeedOffsetProb = 0.15;
            config.messageInterval = 2;
            config.numAttackers = 12;
            break;

        case NORMAL_TRAFFIC:
            config.dosProb = 0.10; //0.15
            config.replayAttackProb = 0.10;
            config.randomSpeedProb = 0.10; //0.15
            config.randomPositionProb = 0.15;
            config.positionProb = 0.25;     // Increased 0.30
            config.constantSpeedOffsetProb = 0.15;
            config.messageInterval = 2;
            config.numAttackers = 8;
            break;
    }
    return config;
}


void VeinsInetSampleApplication::rotateScenario() {
    double currentTime = simTime().dbl();
    double scenarioDuration = 45;  // Shorter duration for each scenario

    if (currentTime - lastScenarioChange >= scenarioDuration) {
        currentScenario = static_cast<ScenarioType>((static_cast<int>(currentScenario) + 1) % 3);
        lastScenarioChange = currentTime;
        EV_INFO << "Rotating to scenario: " << currentScenario
                << " at time " << currentTime
                << " of " << scenarioDuration << "s duration" << endl;
    }
}


bool VeinsInetSampleApplication::isConnectionBlocked(const std::string& sender, const std::string& receiver) const {
    std::string connection = sender + "->" + receiver;
    return blockedConnections.find(connection) != blockedConnections.end();
}

void VeinsInetSampleApplication::updateBlockedConnections(const std::string& sender, const std::string& receiver, bool should_block) {
    std::string connection = sender + "->" + receiver;
    if (should_block) {
        blockedConnections.insert(connection);
        EV_INFO << "Blocking connection: " << connection << endl;
    } else {
        blockedConnections.erase(connection);
        EV_INFO << "Unblocking connection: " << connection << endl;
    }
}

bool VeinsInetSampleApplication::shouldProcessMessage(const std::string& sender, const std::string& receiver) const {
    return !isConnectionBlocked(sender, receiver);
}

// ___________________________________ get the message _________________________________ //
void VeinsInetSampleApplication::processPacket(std::shared_ptr<inet::Packet> pk) {
    auto payload = pk->peekAtFront<VeinsInetSampleMessage>();

    std::string senderId = payload->getSenderId();
    std::string receiverId = getParentModule()->getFullName();

    if (senderId == receiverId) {
        return;
    }

    // Check if connection is blocked before processing
    if (!shouldProcessMessage(senderId, receiverId)) {
        EV_INFO << "Message dropped - connection is blocked by GRL" << endl;
        return;
    }

    // Collect all features
    MessageData msgData = {};

    // Fill the structure
    strncpy(msgData.sender, payload->getSenderId(), 99);
    strncpy(msgData.receiverId, getParentModule()->getFullName(), 99);
    msgData.posx = payload->getPosx();
    msgData.posy = payload->getPosy();
    msgData.spdx = payload->getSpdx();
    msgData.spdy = payload->getSpdy();
    msgData.aclx = payload->getAclx();
    msgData.acly = payload->getAcly();
    msgData.hedx = payload->getHedx();
    msgData.hedy = payload->getHedy();
    msgData.sendTime = simTime().dbl();
    msgData.label = payload->getMalicious() ? 1 : 0;
    strncpy(msgData.attackType, payload->getMalicious() ? payload->getAttackType() : "normal_behavior", 49);

    // Add this before sending data
    EV_INFO << "Sending data: size=" << sizeof(msgData)
            << ", senderId=" << msgData.sender
            << ", receiverId=" << msgData.receiverId << endl;


    // Create socket
    SOCKET sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        EV_ERROR << "Socket creation error: " << WSAGetLastError() << endl;
        return;
    }

    // Setup server address
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PYTHON_SERVER_PORT);

    unsigned long addr = inet_addr(PYTHON_SERVER_IP);
    if (addr == INADDR_NONE) {
        EV_ERROR << "Invalid IP address" << endl;
        closesocket(sock);
        return;
    }
    serv_addr.sin_addr.s_addr = addr;

    // Connect to Python server
    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) == SOCKET_ERROR) {
        EV_ERROR << "Connection Failed: " << WSAGetLastError() << endl;
        closesocket(sock);
        return;
    }

    // Send all data
    if (::send(sock, (char*)&msgData, sizeof(msgData), 0) == SOCKET_ERROR) {
        EV_ERROR << "Send failed: " << WSAGetLastError() << endl;
        closesocket(sock);
        return;
    }

    // Receive both prediction and GRL decision
    int response[2];  // [prediction, should_prune]
    if (recv(sock, (char*)&response, sizeof(response), 0) == SOCKET_ERROR) {
        EV_ERROR << "Receive failed: " << WSAGetLastError() << endl;
        closesocket(sock);
        return;
    }

    int prediction = response[0];
    int should_prune = response[1];

    // Update blocked connections based on GRL decision
    updateBlockedConnections(senderId, receiverId, should_prune == 1);

    // Log results
    EV_INFO << "Message from " << senderId << " to " << receiverId
            << " classified as: " << (prediction == 1 ? "Malicious" : "Normal")
            << ", GRL decision: " << (should_prune == 1 ? "Prune" : "Maintain") << endl;

    closesocket(sock);


    // Log to file
        std::ofstream logFile;
        bool isNewFile = !std::ifstream("C:/Users/Latifa/src/vanetTuto/simulations/veins_inet_openStreetMap/AD/data/real_time_logging.csv");

        logFile.open("C:/Users/Latifa/src/vanetTuto/simulations/veins_inet_openStreetMap/AD/data/real_time_logging.csv", std::ios::app);
        if (isNewFile) {
            logFile << "sendTime,SenderID,ReceiverID,PosX,PosY,SpdX,SpdY,AclX,AclY,HedX,HedY,AttackType,Status,Prediction,GRLAction\n";
        }

        logFile << msgData.sendTime << ","
                << msgData.sender << ","
                << msgData.receiverId << ","
                << msgData.posx << ","
                << msgData.posy << ","
                << msgData.spdx << ","
                << msgData.spdy << ","
                << msgData.aclx << ","
                << msgData.acly << ","
                << msgData.hedx << ","
                << msgData.hedy << ","
                << msgData.attackType << ","
                << msgData.label << ","
                << prediction << ","
                << should_prune << "\n";
        logFile.close();

    // Call Python script to log messgs
    std::string command = "python3 C:/Users/Latifa/src/vanetTuto/simulations/veins_inet_openstreetMap/python_openstreetMap.py " +
            std::string(msgData.sender) + " " +
            std::string(msgData.receiverId) + " " +
            std::to_string(msgData.posx) + " " +
            std::to_string(msgData.posy) + " " +
            std::to_string(msgData.spdx) + " " +
            std::to_string(msgData.spdy) + " " +
            std::to_string(msgData.aclx) + " " +
            std::to_string(msgData.acly) + " " +
            std::to_string(msgData.hedx) + " " +
            std::to_string(msgData.hedy) + " " +
            std::to_string(msgData.label) + " " +
            std::string(msgData.attackType) + " " +
            std::to_string(msgData.sendTime);

      system(command.c_str());

    EV_INFO << "Message from " << senderId << " to " << receiverId
            << " classified as: " << (prediction == 1 ? "Malicious" : "Normal")
            << ", GRL decision: " << (should_prune == 1 ? "Prune" : "Keep") << endl;
}


//______________________ Initialization and cleanup ___________________________________//
void VeinsInetSampleApplication::initialize(int stage) {
    veins::VeinsInetApplicationBase::initialize(stage);

    // Initialize random seed
    srand(static_cast<unsigned int>(std::time(nullptr)) + getParentModule()->getIndex());

    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        EV_ERROR << "WSAStartup failed" << endl;
        return;
    }
}

void VeinsInetSampleApplication::finish() {
    WSACleanup();
}

bool VeinsInetSampleApplication::stopApplication() {
    // perform any necessary cleanup
    WSACleanup();
    return true;
}
