# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from pxr import Plug, Usd, UsdPhysics

import newton_usd_schemas  # noqa: F401

USD_HAS_LIMITS = Usd.GetVersion() >= (0, 25, 11)


class TestNewtonCollisionPipelineAPI(unittest.TestCase):
    def setUp(self):
        self.stage: Usd.Stage = Usd.Stage.CreateInMemory()
        self.scene: Usd.Prim = UsdPhysics.Scene.Define(self.stage, "/Scene").GetPrim()

    def test_api_registered(self):
        plug_type = Plug.Registry().FindTypeByName("NewtonPhysicsCollisionPipelineAPI")
        self.assertEqual(plug_type.typeName, "NewtonPhysicsCollisionPipelineAPI")
        schema_type = Usd.SchemaRegistry().GetSchemaTypeName("NewtonPhysicsCollisionPipelineAPI")
        self.assertEqual(schema_type, "NewtonCollisionPipelineAPI")

    def test_api_application(self):
        self.assertTrue(self.scene.CanApplyAPI("NewtonCollisionPipelineAPI"))
        self.scene.ApplyAPI("NewtonCollisionPipelineAPI")
        self.assertTrue(self.scene.HasAPI("NewtonSceneAPI"))
        self.assertTrue(self.scene.HasAPI("NewtonCollisionPipelineAPI"))
        self.assertTrue(self.scene.HasAttribute("newton:maxSolverIterations"))
        self.assertTrue(self.scene.HasAttribute("newton:collisionPipeline:broadPhase"))

    def test_api_limitations(self):
        prim: Usd.Prim = self.stage.DefinePrim("/NotScene", "Xform")
        self.assertFalse(prim.CanApplyAPI("NewtonCollisionPipelineAPI"))

    def test_broad_phase(self):
        self.scene.ApplyAPI("NewtonCollisionPipelineAPI")
        attr = self.scene.GetAttribute("newton:collisionPipeline:broadPhase")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), "explicit")

        self.assertTrue(attr.Set("sap"))
        self.assertTrue(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), "sap")
        self.assertEqual(set(attr.GetMetadata("allowedTokens")), {"nxn", "sap", "explicit"})

    def test_max_triangle_pairs(self):
        self.scene.ApplyAPI("NewtonCollisionPipelineAPI")
        attr = self.scene.GetAttribute("newton:collisionPipeline:maxTrianglePairs")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), 1000000)

        self.assertTrue(attr.Set(500000))
        self.assertTrue(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), 500000)

        if USD_HAS_LIMITS:
            hard = attr.GetHardLimits()
            self.assertTrue(hard.IsValid())
            self.assertEqual(hard.GetMinimum(), 1)
            self.assertIsNone(hard.GetMaximum())

    def test_rigid_contact_max(self):
        self.scene.ApplyAPI("NewtonCollisionPipelineAPI")
        attr = self.scene.GetAttribute("newton:collisionPipeline:rigidContactMax")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), -1)

        self.assertTrue(attr.Set(2048))
        self.assertTrue(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), 2048)

        if USD_HAS_LIMITS:
            hard = attr.GetHardLimits()
            self.assertTrue(hard.IsValid())
            self.assertEqual(hard.GetMinimum(), -1)
            self.assertIsNone(hard.GetMaximum())

    def test_reduce_contacts(self):
        self.scene.ApplyAPI("NewtonCollisionPipelineAPI")
        attr = self.scene.GetAttribute("newton:collisionPipeline:reduceContacts")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), True)

        self.assertTrue(attr.Set(False))
        self.assertTrue(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), False)

    def test_soft_contact_max(self):
        self.scene.ApplyAPI("NewtonCollisionPipelineAPI")
        attr = self.scene.GetAttribute("newton:collisionPipeline:softContactMax")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), -1)

        self.assertTrue(attr.Set(4096))
        self.assertTrue(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), 4096)

        if USD_HAS_LIMITS:
            hard = attr.GetHardLimits()
            self.assertTrue(hard.IsValid())
            self.assertEqual(hard.GetMinimum(), -1)
            self.assertIsNone(hard.GetMaximum())

    def test_soft_contact_gap(self):
        self.scene.ApplyAPI("NewtonCollisionPipelineAPI")
        attr = self.scene.GetAttribute("newton:collisionPipeline:softContactGap")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), -math.inf)

        self.assertTrue(attr.Set(0.02))
        self.assertTrue(attr.HasAuthoredValue())
        self.assertAlmostEqual(attr.Get(), 0.02)

        if USD_HAS_LIMITS:
            hard = attr.GetHardLimits()
            self.assertTrue(hard.IsValid())
            self.assertAlmostEqual(hard.GetMinimum(), 0.0)
            self.assertIsNone(hard.GetMaximum())

    def test_enable_rigid_soft_full_surface_contact(self):
        self.scene.ApplyAPI("NewtonCollisionPipelineAPI")
        attr = self.scene.GetAttribute("newton:collisionPipeline:enableRigidSoftFullSurfaceContact")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), False)

        self.assertTrue(attr.Set(True))
        self.assertTrue(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), True)

    def test_requires_grad(self):
        self.scene.ApplyAPI("NewtonCollisionPipelineAPI")
        attr = self.scene.GetAttribute("newton:collisionPipeline:requiresGrad")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), "inherit")

        self.assertTrue(attr.Set("true"))
        self.assertTrue(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), "true")
        self.assertEqual(set(attr.GetMetadata("allowedTokens")), {"inherit", "true", "false"})

    def test_deterministic(self):
        self.scene.ApplyAPI("NewtonCollisionPipelineAPI")
        attr = self.scene.GetAttribute("newton:collisionPipeline:deterministic")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), False)

        self.assertTrue(attr.Set(True))
        self.assertTrue(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), True)


if __name__ == "__main__":
    unittest.main()
