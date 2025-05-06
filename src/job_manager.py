import time
import subprocess

class JobManager:
    @staticmethod
    def check_job_completion(job_ids):
        """Checks if the submitted jobs have completed using 'squeue'."""
        print("Checking job status with squeue...")
        completed_jobs = set()

        while len(completed_jobs) < len(job_ids):
            for job_id in job_ids:
                if job_id in completed_jobs:
                    continue  # Skip already completed jobs

                # Check if the job is still in the queue with 'squeue'
                result = subprocess.run(["squeue", "--job", job_id], capture_output=True, text=True)

                if job_id not in result.stdout:
                    # If the job is not in 'squeue', it is completed or failed
                    completed_jobs.add(job_id)
                    print(f"Job {job_id} no longer in 'squeue'. Assuming completed.")
                else:
                    print(f"Job {job_id} is still running.")

            if len(completed_jobs) < len(job_ids):
                print(f"Waiting for jobs to finish... Checking again in 3 minutes.")
                time.sleep(180)  # Check every 3 minutes

        print("All jobs have finished.")
