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

#ifndef MJPC_TASKS_MOTORCYCLE_MOTORCYCLE_H_
#define MJPC_TASKS_MOTORCYCLE_MOTORCYCLE_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class Motorcycle : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Motorcycle* task) : mjpc::BaseResidualFn(task) {}

// ----------------- Residuals for motorcycle task ----------------
//   Number of residuals: 7
//     Residual (0-4): control
//     Residual (5-6): XY displacement between nose and target
// -------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  Motorcycle() : residual_(this) {}
// -------- Transition for motorcycle task --------
//   If motorcycle is within tolerance of goal ->
//   move goal randomly.
// ---------------------------------------------
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_MOTORCYCLE_MOTORCYCLE_H_
