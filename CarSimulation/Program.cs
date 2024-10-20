using System;
using System.Threading;

public class Car
{
    private int speed = 0;
    private double wheelAngle = 0;
    private bool simulatorRunning = true;
    private bool running = false;
    private bool onTheRoad = false;
    private bool onTheHighway = false;
    private int eventsHandled = 0;

    public void Act(string command)
    {
        switch (command)
        {
            case "start the engine":
                StartTheEngine();
                break;
            case "drive":
                Drive();
                break;
            case "turn":
                Turn();
                break;
            case "accelerate":
                Accelerate();
                break;
            case "brake":
                Brake();
                break;
            case "obstacle":
                AvoidObstacle();
                break;
            case "highway":
                Highway();
                break;
            case "exit highway":
                ExitHighway();
                break;
            case "overtake":
                Overtake();
                break;
            case "truck":
                Truck();
                break;
            case "status":
                Status();
                break;
            case "stop":
                Stop();
                break;
            case "exit":
                Exit();
                break;
            case "help":
            case "?":
                Console.WriteLine("Available commands: start the engine, drive, turn, accelerate, brake, obstacle, highway, exit highway, overtake, truck, status, stop, exit, help");
                break;
            default:
                Console.WriteLine("Unknown command");
                break;
        }

        eventsHandled++;
    }

    private bool NotRunning()
    {
        if (!running)
        {
            Console.WriteLine("Car's engine is not running. Please start the engine first.");
        }
        return !running;
    }

    private bool NotOnTheRoad()
    {
        if (!onTheRoad)
        {
            Console.WriteLine("Car is not on the road. Please enter a \"drive\" command first.");
        }
        return !onTheRoad;
    }

    private void StartTheEngine()
    {
        if (NotRunning())
        {
            Console.WriteLine("Starting the engine...");
            Thread.Sleep(1000);
            Console.WriteLine("Engine started. Car is ready to drive.");
            Thread.Sleep(1000);
            Console.WriteLine("Please enter a \"drive\" command to start driving.");
            running = true;
        }
        else if (onTheRoad || onTheHighway)
        {
            Console.WriteLine("Engine is already running. You can continue driving.");
        }
        else
        {
            Console.WriteLine("Engine is already running. Please enter a \"drive\" command to start driving.");
        }
    }

    private void Drive()
    {
        if (NotRunning()) return;
        Console.WriteLine("Driving...");
        onTheRoad = true;
        Thread.Sleep(1000);
        Accelerate(5);
        Thread.Sleep(1000);
        Console.WriteLine("Car is on the road.");
    }

    private void Turn()
    {
        if (NotRunning()) return;
        Random random = new Random();
        double angle = random.Next(-30, 31);
        for (int i = 0; i < 3; i++)
        {
            wheelAngle = angle;
            Thread.Sleep(100);
            Console.WriteLine($"Turning... Wheel angle: {wheelAngle}°");
            angle -= angle >= 0 ? 7.50 : -7.50;
        }
        wheelAngle = 0; // Reset to forward after the turn
    }

    private void Accelerate(int duration = 3)
    {
        if (NotRunning()) return;
        if (NotOnTheRoad()) return;
        for (int i = 0; i < duration; i++)
        {
            if (onTheHighway)
            {
                if (speed < 140) // Max speed
                {
                    speed += 10;
                    Thread.Sleep(100);
                    Console.WriteLine($"Accelerating... Speed: {speed} km/h");
                }
            }
            else
            {
                if (speed < 50)
                {
                    speed += 10;
                    Thread.Sleep(100);
                    Console.WriteLine($"Accelerating... Speed: {speed} km/h");
                }
            }
        }
    }

    private void Brake(int duration = 2)
    {
        if (!running || speed == 0)
        {
            Console.WriteLine("Car is already stopped.");
            return;
        }
        for (int i = 0; i < duration; i++)
        {
            if (speed > 0)
            {
                speed -= 10;
            }
            else
            {
                speed = 0;
            }
            Thread.Sleep(100);
            Console.WriteLine($"Braking... Speed: {speed} km/h");
        }
    }

    private void AvoidObstacle()
    {
        if (NotRunning()) return;
        if (speed == 0)
        {
            Console.WriteLine("Car is already stopped.");
            return;
        }
        Console.WriteLine("Obstacle detected. Avoiding the obstacle...");
        Thread.Sleep(1000);
        Brake(1);
        Thread.Sleep(1000);
        Turn();
        Console.WriteLine($"Car's speed decreasing to {speed}");
    }

    private void Highway()
    {
        if (NotRunning()) return;
        if (!onTheRoad)
        {
            Console.WriteLine("Car is not on the road. Please enter a \"drive\" command first.");
            return;
        }
        if (onTheHighway)
        {
            Console.WriteLine("Car is already on the highway.");
            return;
        }

        Console.WriteLine("Entering the highway...");
        Thread.Sleep(1000);
        Console.WriteLine("Car is on the highway.");
        Thread.Sleep(1000);
        Accelerate(10);
        wheelAngle = 0;
        onTheHighway = true;
    }

    private void ExitHighway()
    {
        if (NotRunning()) return;
        if (!onTheHighway)
        {
            Console.WriteLine("Car is not on the highway.");
            return;
        }
        Console.WriteLine("Exiting the highway...");
        Thread.Sleep(1000);
        Brake(9);
        Console.WriteLine("Car is on the road.");
        Thread.Sleep(1000);
        onTheHighway = false;
    }

    private void Overtake()
    {
        if (NotRunning()) return;
        if (NotOnTheRoad()) return;
        if (onTheHighway)
        {
            Console.WriteLine("Overtaking... Speed set to 160 km/h.");
            speed = 160;
        }
        else
        {
            Console.WriteLine("Overtaking... Speed set to 70 km/h.");
            speed = 70;
        }
    }

    private void Truck()
    {
        if (NotRunning()) return;
        if (NotOnTheRoad())
        {
            Console.WriteLine("Car is not on the road. Please enter a \"drive\" command first.");
            return;
        }
        if (onTheHighway)
        {
            Console.WriteLine("Truck ahead! Slowing down to 100 km/h.");
            speed = 100;
        }
        else
        {
            speed = 50;
            Console.WriteLine("Truck ahead! Slowing down to 50 km/h.");
        }
    }

    private void Status()
    {
        Console.WriteLine($"Car's current status: speed = {speed}, wheel angle = {wheelAngle}°, Engine = {(running ? "running" : "not running")}");
        if (onTheHighway)
        {
            Console.WriteLine("Car is on the highway.");
        }
        else if (onTheRoad)
        {
            Console.WriteLine("Car is on the road.");
        }
        else
        {
            Console.WriteLine("Car is not on the road.");
        }
    }

    private void Stop()
    {
        if (!running)
        {
            Console.WriteLine("Car is already stopped.");
            return;
        }
        if (onTheHighway)
        {
            ExitHighway();
        }
        Console.WriteLine("Stopping the car...");
        Thread.Sleep(1000);
        running = false;
        onTheRoad = false;
        speed = 0;
        Console.WriteLine("Car has stopped.");
        Thread.Sleep(1000);
    }

    private void Exit()
    {
        Console.WriteLine("Exiting the car simulation...");
        Thread.Sleep(1000);
        running = false;
        simulatorRunning = false;
        Console.WriteLine("Car simulation ended.");
        Thread.Sleep(1000);
    }

    public static void Main(string[] args)
    {
        Console.WriteLine("Starting car simulation...");
        Thread.Sleep(1000);
        Console.WriteLine("Car simulation started.");
        Thread.Sleep(1000);

        Car car1 = new Car();
        while (car1.simulatorRunning)
        {
            Console.Write("Enter a command: ");
            string command = Console.ReadLine().Trim().ToLower();
            car1.Act(command);
        }

        Console.WriteLine($"Simulation ended after handling {car1.eventsHandled} events.");
    }
}
