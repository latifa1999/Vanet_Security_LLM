# VANET Co-Simulation Environment

OMNeT++ / SUMO / Veins simulation used to generate V2V traffic, inject
message-level attacks, and apply real-time mitigation decisions returned by the
Python backend.

This is the **simulation component** of the framework described in *"LLM-Guided
Multi-Agent Graph Reinforcement Learning for Anomaly Detection and Adaptive
Mitigation in Vehicular Networks."* The MIST detector and the MAGRL policy live in
separate repositories and run as a Python server that this simulation connects to.

---

## What it does

Vehicles move along a real road network imported from OpenStreetMap and exchange
periodic safety messages over IEEE 802.11p. A configurable fraction of vehicles
behaves maliciously, injecting one of six message-level attacks.

Every transmitted message is forwarded over a TCP socket to the Python backend,
which returns an anomaly classification and a keep/prune decision. The decision is
applied inside the simulation, closing the loop between detection and mitigation.

The simulation runs in two modes:

- **Data generation** — messages and labels are logged to CSV for offline training
  of MIST and MAGRL.
- **Online evaluation** — trained models are queried live and their mitigation
  decisions take effect during the run.

---

## Prerequisites

| Component | Version | Purpose |
|---|---|---|
| [OMNeT++](https://omnetpp.org/download/old) | 5.6.2 | discrete-event simulation framework |
| [SUMO](https://sourceforge.net/projects/sumo/files/sumo/version%201.8.0/) | 1.8.0 | microscopic traffic simulation |
| [Veins](https://veins.car2x.org/download/) | 5.2 | vehicular networking framework |
| [INET](https://inet.omnetpp.org/Download.html) | 4.x | protocol stack used by `veins_inet` |
| Python | 3.10 | backend server (separate repository) |

Versions matter. Veins 5.2 expects OMNeT++ 5.6.x and INET 4.x; mixing major
versions leads to build failures that are tedious to diagnose.

### Setting up

Follow the official Veins tutorial before touching this repository — it covers
launching SUMO from OMNeT++ through TraCI, which everything here depends on:

**https://veins.car2x.org/tutorial/**

Confirm the stock Veins example runs end to end (vehicles moving in the OMNeT++
GUI, driven by SUMO) before continuing. If that does not work, nothing here will.

Make sure `SUMO_HOME` is set and `sumo`, `sumo-gui`, `netconvert`, and `duarouter`
are on your `PATH`.

> **Platform note.** The socket client uses the Winsock API, so the simulation
> side currently builds on Windows. Porting to POSIX sockets requires changes in
> `VeinsInetSampleApplication.cc`.

---

## Repository structure

```
.
├── Road_1_Casablanca/
│   ├── map.osm                 # OSM export the network was built from
│   ├── casa50.net.xml          # SUMO network — shared by all three densities
│   ├── casa50.trips.xml        # 50-vehicle scenario
│   ├── casa50.rou.xml
│   ├── casa50.sumo.cfg
│   ├── casa50.launchd.xml
│   ├── casa100.*               # 100-vehicle scenario
│   ├── casa200.*               # 200-vehicle scenario
│   └── road.jpg                # rendered view of the topology
├── Road_2_Orlando/             # same layout, "orlando" prefix
├── Road_3_Madrid/              # same layout, "madrid" prefix
├── vanet/
│   └── src/
│       ├── veins_inet/
│       │   ├── VeinsInetSampleApplication.cc   # beaconing, attacks, socket client
│       │   ├── VeinsInetSampleApplication.h
│       │   ├── VeinsInetSampleMessage.msg      # message definition
│       │   ├── VeinsInetManager.*              # TraCI manager
│       │   ├── VeinsInetMobility.*
│       │   └── VeinsInetCar.ned
│       ├── Makefile
│       └── makefrag
├── Scenario.ned                # network definition
├── omnetpp.ini                 # simulation parameters
└── package.ned
```

Each road directory holds **one road network** (`*.net.xml`) reused across the
three vehicle densities — only the trips, routes, and configs differ. Density is
encoded in the filename prefix (`casa50`, `casa100`, `casa200`), so the three
densities are strictly comparable: same geometry, different traffic.

| Scenario | Area | Prefix |
|---|---|---|
| Road 1 — Casablanca, Morocco | 12.5 km × 6.9 km | `casa` |
| Road 2 — Orlando, USA | 6.9 km × 9.0 km | `orlando` |
| Road 3 — Madrid, Spain | 4.9 km × 10.8 km | `madrid` |

The generated SUMO files are committed deliberately: they are the exact
experimental conditions the reported results come from. Re-running
`randomTrips.py` produces different traffic and will not reproduce the paper's
numbers.

---

## Building

```bash
cd vanet
make makefiles
make
```

This produces `vanet` (release) and `vanet_dbg` (debug). If linking fails, check
that `makefrag` points at your Veins and INET installation paths.

Compiled binaries are not tracked in git — build them locally.

---

## Generating a road scenario

The scenarios in this repository are already generated; this section documents how
they were produced and how to add a new one.

**The network is built once per road, then reused for every density.** Only steps
3–6 are repeated per vehicle count. Regenerating the network for each density
would produce slightly different road graphs and break the density comparison.

### 1. Export the map (once per road)

Open [OpenStreetMap](https://www.openstreetmap.org), navigate to the area, and use
**Export** to download `map.osm`. Large exports may need the Overpass API mirror
that the export panel offers.

### 2. Convert to a SUMO network (once per road)

```bash
netconvert --osm-files map.osm --output-file casa50.net.xml \
  --geometry.remove --roundabouts.guess --ramps.guess \
  --junctions.join --tls.guess-signals --tls.discard-simple --tls.join
```

These flags clean up raw OSM geometry: merging duplicate junctions, inferring
ramps and roundabouts, and discarding trivial traffic lights that would otherwise
stall traffic.

### 3. Generate trips (per density)

```bash
python $SUMO_HOME/tools/randomTrips.py -n casa.net.xml -e 50  -o casa50.trips.xml
python $SUMO_HOME/tools/randomTrips.py -n casa.net.xml -e 100 -o casa100.trips.xml
python $SUMO_HOME/tools/randomTrips.py -n casa.net.xml -e 200 -o casa200.trips.xml
```

`-e` sets the end time of trip generation, which together with network size
determines the vehicle count. The resulting count depends on the map, so verify it
after routing rather than assuming `-e` equals the number of vehicles.

### 4. Compute routes (per density)

```bash
duarouter -n casa50.net.xml --route-files casa50.trips.xml \
  -o casa50.rou.xml --ignore-errors
```

`--ignore-errors` drops trips with no valid route rather than aborting — common
with OSM exports containing disconnected fragments.

### 5. Create `<prefix>.sumo.cfg` (per density)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="casa50.net.xml"/>
        <route-files value="casa50.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="400"/>
        <step-length value="0.1"/>
    </time>
    <report>
        <no-step-log value="true"/>
    </report>
</configuration>
```

The step length must match the beacon interval set in `omnetpp.ini`.

### 6. Create `<prefix>.launchd.xml` (per density)

Veins uses this to start SUMO through TraCI:

```xml
<?xml version="1.0"?>
<launch>
    <copy file="casa50.net.xml"/>
    <copy file="casa50.rou.xml"/>
    <copy file="casa50.sumo.cfg" type="config"/>
</launch>
```

### 7. Point the simulation at it

In `omnetpp.ini`:

```ini
*.manager.launchConfig = xmldoc("Road_1_Casablanca/casa50.launchd.xml")
```

---

## Simulation parameters

Values used in the paper, configured in `omnetpp.ini`:

| Parameter | Value |
|---|---|
| Simulation time | 400 s |
| Beacon interval | 0.1 s |
| Number of vehicles | 50 / 100 / 200 |
| Attacker ratio | 10% / 30% / 50% |
| Attack types | 6 (see below) |
| MAC protocol | IEEE 802.11p |
| Frequency band | 5.9 GHz |
| Bandwidth | 10 MHz |
| Transmission power | 20 mW |
| Communication range | 400 m |

Each scenario was repeated with different random seeds.

### Attack types

| Category | Attack | Effect |
|---|---|---|
| Volume-based | DoS Disruptive | floods the channel with falsified safety messages |
| Data corruption | Constant Position Offset | fixed offset applied to reported position |
| | Random Position | position randomly modified |
| | Constant Speed Offset | fixed offset applied to reported speed |
| | Random Speed | speed randomly perturbed |
| Replay | Replay Attack | re-transmits previously legitimate messages |

Attack injection is implemented in `VeinsInetSampleApplication.cc`.

---

## Running

### 1. Start the Python backend first

The simulation connects on every transmission and blocks waiting for a reply, so
the server must be listening before the run starts.

```bash
# in the detection/mitigation repository
python realtime_ad_magrl_server.py
```

Wait for `Server listening on 127.0.0.1:5000`.

### 2. Launch the simulation

```bash
cd vanet
./vanet -u Cmdenv -c <ConfigName> -n .:../src:$INET_ROOT/src omnetpp.ini
```

Or open the project in the OMNeT++ IDE and run the desired configuration from
`omnetpp.ini`. Use `Cmdenv` for batch runs and the Qtenv GUI when you want to watch
vehicle movement and message exchange.

---

## Socket protocol

The simulation opens a TCP connection to `127.0.0.1:5000` for each transmitted
message.

**Request — 326 bytes:**

| Offset | Size | Field |
|---|---|---|
| 0 | 100 | `sender_id` (null-terminated string) |
| 100 | 100 | `receiver_id` (null-terminated string) |
| 200 | 80 | 10 doubles: posX, posY, spdX, spdY, aclX, aclY, hedX, hedY, simTime, reserved |
| 280 | 42 | `attack_type` (null-terminated string) |
| 322 | 4 | `is_malicious` (int, ground truth — logging and evaluation only) |

**Response — 8 bytes:** two ints, `(ml_prediction, should_prune)`.
`should_prune = 1` removes the link; `0` keeps it.

Ground truth is transmitted so the backend can log and evaluate. It is never used
as model input.

---

## Output

Data-generation runs produce a CSV with one row per message:

```
sendTime, SenderID, ReceiverID,
posx, posy, spdx, spdy, aclx, acly, hedx, hedy,
AttackType, label
```

This is the input to MIST training. After running inference, the `Prediction`,
`Prob_Normal`, and `Prob_Malicious` columns are appended, producing the file MAGRL
consumes.

---

## Troubleshooting

**SUMO fails to start from OMNeT++** — check `SUMO_HOME`, and that the paths in
`<prefix>.launchd.xml` are relative to the config file.

**No vehicles appear** — usually an empty or invalid `.rou.xml`. Run SUMO standalone
first: `sumo-gui -c casa50.sumo.cfg`.

**Simulation hangs at the first transmission** — the Python backend is not running
or is on a different port. The socket call blocks until it receives a reply.

**Build errors after switching Veins or INET versions** — run `make clean` and
regenerate makefiles; stale object files are a frequent cause.

---