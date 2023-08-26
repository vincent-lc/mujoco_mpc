// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/bike/bike.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string Bike::XmlPath() const {
  return GetModelPath("bike/task.xml");
}
std::string Bike::Name() const { return "Bike"; }
// ----------------- Residuals for bike task ----------------
//   Number of residuals: 7
//     Residual (0-4): control
//     Residual (5-6): XY displacement between nose and target
// -------------------------------------------------------------
void Bike::ResidualFn::Residual(const mjModel* model, const mjData* data,
                       double* residual) const {
  // ---------- Residual (0-4) ----------
  // controls
  mju_copy(residual, data->ctrl, model->nu);

  // ---------- Residuals (5-6) ----------
  // nose to target XY displacement
  double* target = SensorByName(model, data, "target");
  double* nose = SensorByName(model, data, "nose");
  mju_sub(residual + model->nu, nose, target, 2);
}

// -------- Transition for bike task --------
//   If bike is within tolerance of goal ->
//   move goal randomly.
// ---------------------------------------------
void Bike::TransitionLocked(mjModel* model, mjData* data) {
  double* target = SensorByName(model, data, "target");
  double* nose = SensorByName(model, data, "nose");
  double nose_to_target[2];
  mju_sub(nose_to_target, target, nose, 2);
  if (mju_norm(nose_to_target, 2) < 0.04) {
    absl::BitGen gen_;
    data->mocap_pos[0] = absl::Uniform<double>(gen_, -.8, .8);
    data->mocap_pos[1] = absl::Uniform<double>(gen_, -.8, .8);
  }
}

}  // namespace mjpc
