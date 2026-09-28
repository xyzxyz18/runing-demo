import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as web_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.jobs_patch = patch.object(web_app, "JOBS_DIR", Path(self.temp_dir.name))
        self.jobs_patch.start()
        web_app.jobs.clear()
        self.client = web_app.app.test_client()

    def tearDown(self):
        self.jobs_patch.stop()
        self.temp_dir.cleanup()

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"videoInput", response.data)
        self.assertIn(b"exportPdfButton", response.data)

    def test_pdf_artifact_can_be_downloaded(self):
        job_id = "pdf-job"
        folder = Path(self.temp_dir.name) / job_id
        folder.mkdir()
        (folder / "report.pdf").write_bytes(b"%PDF-1.4\n")
        web_app.jobs[job_id] = {"id": job_id, "state": "completed"}
        response = self.client.get(f"/results/{job_id}/report.pdf")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")
        response.close()

    def test_rejects_missing_or_invalid_video(self):
        self.assertEqual(self.client.post("/api/jobs").status_code, 400)
        response = self.client.post(
            "/api/jobs",
            data={"video": (io.BytesIO(b"not a video"), "sample.txt")},
        )
        self.assertEqual(response.status_code, 400)

    def test_realtime_endpoint_requires_jpeg(self):
        response = self.client.post("/api/realtime/pose", data=b"frame")
        self.assertEqual(response.status_code, 415)

    @patch.object(web_app.executor, "submit")
    def test_accepts_video_and_queues_job(self, submit):
        response = self.client.post(
            "/api/jobs",
            data={"video": (io.BytesIO(b"video bytes"), "sample.mp4")},
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.get_json()["id"]
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").get_json()["state"], "queued")
        history = self.client.get("/api/jobs").get_json()["jobs"]
        self.assertEqual([item["id"] for item in history], [job_id])
        self.assertTrue(any((Path(self.temp_dir.name) / job_id).glob("*.mp4")))
        source_response = self.client.get(f"/results/{job_id}/source.mp4")
        self.assertEqual(source_response.status_code, 200)
        source_response.close()
        submit.assert_called_once()

    @patch.object(web_app.executor, "submit")
    def test_reanalyzes_from_preserved_source(self, submit):
        created = self.client.post(
            "/api/jobs",
            data={"video": (io.BytesIO(b"video bytes"), "sample.mp4")},
        ).get_json()
        response = self.client.post(f"/api/jobs/{created['id']}/reanalyze")
        self.assertEqual(response.status_code, 202)
        self.assertNotEqual(response.get_json()["id"], created["id"])
        self.assertEqual(submit.call_count, 2)


if __name__ == "__main__":
    unittest.main()
