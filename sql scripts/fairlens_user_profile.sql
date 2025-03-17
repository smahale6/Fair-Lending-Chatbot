-- Check if the table already exists, and drop it if it does
IF OBJECT_ID('dbo.fairlens_user_profile', 'U') IS NOT NULL
    DROP TABLE dbo.fairlens_user_profile;
GO

-- Create the table with UserName as the Primary Key
CREATE TABLE dbo.fairlens_user_profile (
    UserName VARCHAR(50) PRIMARY KEY,         -- Primary key for unique usernames
    Password VARCHAR(256) NOT NULL,           -- Encrypted password (hashed before storing)
    Title VARCHAR(50),                        -- Job title
    FirstName VARCHAR(50) NOT NULL,           -- User's first name
    LastName VARCHAR(50) NOT NULL,            -- User's last name
    Permissions int,                          -- User permissions (e.g., 1,0)
    EmailAddress VARCHAR(100) NOT NULL        -- Email address
);
GO

-- Add a unique constraint on the Email_Address column to prevent duplicates
ALTER TABLE dbo.fairlens_user_profile
ADD CONSTRAINT UQ_EmailAddress UNIQUE (EmailAddress);
GO


INSERT INTO dbo.fairlens_user_profile 
(UserName, Password, Title, FirstName, LastName, Permissions, EmailAddress)
VALUES 
('c6400', 'hello', 'Mr', 'Shrikanth', 'Mahale', 1, 'shrikanth.mahale@gmail.com');
